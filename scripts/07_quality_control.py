#!/usr/bin/env python3
"""Analisa a qualidade e consolida a base analítica do PropertyValuation.

Entradas:
    data/processed/itbi_transactions_deflated.parquet
    data/processed/census_sector_profile.parquet

Saídas:
    data/processed/itbi_transactions_qc.parquet
    data/processed/itbi_analysis_sample.parquet
    data/processed/quality_control_summary.csv
    data/processed/outlier_thresholds.csv
    data/processed/quality_control_metadata.json

Metodologia:
- processa a associação do ITBI ao perfil socioeconômico do setor censitário;
- analisa consistência de datas, valores, áreas, geocodificação e Censo;
- analisa outliers em escala logarítmica com IQR conservador por ano;
- preserva todas as observações na base auditada;
- processa flags explícitas para diferentes usos analíticos;
- não remove observações silenciosamente.

Uso:
    python scripts/07_quality_control.py

Dependências:
    numpy
    pandas
    pyarrow
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ITBI = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_deflated.parquet"
)
DEFAULT_CENSUS = (
    PROJECT_ROOT / "data" / "processed" / "census_sector_profile.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_qc.parquet"
)
DEFAULT_SAMPLE = (
    PROJECT_ROOT / "data" / "processed" / "itbi_analysis_sample.parquet"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "data" / "processed" / "quality_control_summary.csv"
)
DEFAULT_THRESHOLDS = (
    PROJECT_ROOT / "data" / "processed" / "outlier_thresholds.csv"
)
DEFAULT_METADATA = (
    PROJECT_ROOT / "data" / "processed" / "quality_control_metadata.json"
)

DEFAULT_IQR_MULTIPLIER = 3.0
DEFAULT_MIN_GROUP_SIZE = 50

REQUIRED_ITBI_COLUMNS = {
    "transaction_id",
    "source_year",
    "base_de_calculo",
    "base_de_calculo_real",
    "area_referencia_m2",
    "valor_m2_base",
    "valor_m2_base_real",
    "sector_final_code",
}

CENSUS_METRIC_COLUMNS = [
    "populacao_total",
    "domicilios_total",
    "domicilios_particulares",
    "domicilios_particulares_ocupados",
    "media_moradores_domicilio",
    "densidade_pop_km2",
    "pct_homens",
    "pct_mulheres",
    "pop_0_14",
    "pop_15_29",
    "pop_30_59",
    "pop_60_mais",
    "pct_pop_0_14",
    "pct_pop_15_29",
    "pct_pop_30_59",
    "pct_pop_60_mais",
    "idade_media_aprox",
    "razao_dependencia_0_14_60mais",
    "renda_media_responsavel_com_rendimento_rs",
    "dp_renda_responsavel_rs",
    "cv_renda_responsavel",
]


def utc_now_iso() -> str:
    """Processa o instante atual em UTC para os metadados da execução."""
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    """Processa caminhos relativos sempre a partir da raiz do projeto."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_sector_code(series: pd.Series) -> pd.Series:
    """Processa códigos de setor como strings de até 15 dígitos."""
    return (
        series.astype("string")
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 15)
        .replace("", pd.NA)
    )


def ensure_boolean(
    frame: pd.DataFrame,
    column: str,
    default: bool = False,
) -> pd.Series:
    """Processa uma coluna booleana existente ou cria um padrão auditável."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="boolean")

    return frame[column].astype("boolean").fillna(default)


def load_itbi(path: Path) -> pd.DataFrame:
    """Processa a base monetariamente corrigida e valida seu contrato mínimo."""
    if not path.exists():
        raise FileNotFoundError(
            f"Base do ITBI corrigida não encontrada: {path}. "
            "Execute antes: python scripts/06_deflate_prices.py"
        )

    frame = pd.read_parquet(path)
    missing = REQUIRED_ITBI_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Base do ITBI sem colunas obrigatórias: {sorted(missing)}"
        )

    if frame["transaction_id"].duplicated().any():
        duplicated = int(frame["transaction_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"Base do ITBI contém {duplicated} linhas com transaction_id duplicado."
        )

    return frame


def load_census(path: Path) -> pd.DataFrame:
    """Processa o perfil censitário e seleciona somente atributos analíticos."""
    if not path.exists():
        raise FileNotFoundError(
            f"Perfil censitário não encontrado: {path}. "
            "Execute antes: python scripts/05_prepare_census.py"
        )

    frame = pd.read_parquet(path)
    if "CD_SETOR" not in frame.columns:
        raise ValueError("Perfil censitário sem a coluna CD_SETOR.")

    frame = frame.copy()
    frame["sector_join_key"] = normalize_sector_code(frame["CD_SETOR"])

    # Analisa duplicidades porque o join exige uma linha por setor censitário.
    duplicated = frame["sector_join_key"].duplicated(keep=False)
    if duplicated.any():
        count = int(duplicated.sum())
        raise ValueError(
            f"Perfil censitário contém {count} linhas com setor duplicado."
        )

    selected = ["sector_join_key"]

    if "CD_BAIRRO" in frame.columns:
        selected.append("CD_BAIRRO")
    if "NM_BAIRRO" in frame.columns:
        selected.append("NM_BAIRRO")

    selected.extend(
        column for column in CENSUS_METRIC_COLUMNS if column in frame.columns
    )

    result = frame[selected].copy()
    rename_map = {
        "CD_BAIRRO": "census_bairro_code",
        "NM_BAIRRO": "census_bairro_nome",
    }
    return result.rename(columns=rename_map)


def enrich_with_census(
    itbi: pd.DataFrame,
    census: pd.DataFrame,
) -> pd.DataFrame:
    """Processa a associação do ITBI ao perfil socioeconômico do setor."""
    result = itbi.copy()
    result["sector_join_key"] = normalize_sector_code(result["sector_final_code"])

    result = result.merge(
        census,
        on="sector_join_key",
        how="left",
        validate="many_to_one",
    )

    census_reference = next(
        (
            column
            for column in (
                "populacao_total",
                "domicilios_total",
                "media_moradores_domicilio",
            )
            if column in result.columns
        ),
        None,
    )

    if census_reference is None:
        result["flag_census_matched"] = False
    else:
        result["flag_census_matched"] = result[census_reference].notna()

    return result


def add_structural_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Analisa regras estruturais sem aplicar filtros estatísticos."""
    result = frame.copy()

    base_nominal = pd.to_numeric(result["base_de_calculo"], errors="coerce")
    base_real = pd.to_numeric(result["base_de_calculo_real"], errors="coerce")
    area = pd.to_numeric(result["area_referencia_m2"], errors="coerce")
    value_m2_nominal = pd.to_numeric(result["valor_m2_base"], errors="coerce")
    value_m2_real = pd.to_numeric(result["valor_m2_base_real"], errors="coerce")

    result["flag_base_nominal_invalida"] = base_nominal.isna() | base_nominal.le(0)
    result["flag_base_real_invalida"] = base_real.isna() | base_real.le(0)
    result["flag_area_invalida"] = area.isna() | area.le(0)
    result["flag_valor_m2_nominal_invalido"] = (
        value_m2_nominal.isna() | value_m2_nominal.le(0)
    )
    result["flag_valor_m2_real_invalido"] = (
        value_m2_real.isna() | value_m2_real.le(0)
    )

    # Analisa o percentual transmitido somente quando o campo está preenchido.
    if "perc_transmitido" in result.columns:
        transmitted = pd.to_numeric(result["perc_transmitido"], errors="coerce")
        result["flag_percentual_transmitido_invalido"] = (
            transmitted.notna() & (transmitted.le(0) | transmitted.gt(100))
        )
    else:
        result["flag_percentual_transmitido_invalido"] = False

    # Processa flags herdadas das etapas anteriores para uma interface única.
    result["flag_pago_qc"] = ensure_boolean(result, "flag_pago")
    result["flag_cancelado_qc"] = ensure_boolean(result, "flag_cancelado")
    result["flag_duplicado_qc"] = ensure_boolean(
        result,
        "flag_duplicado_exato",
    )
    result["flag_multiunit_qc"] = ensure_boolean(result, "flag_multiunit")
    result["flag_amostra_mercado_origem"] = ensure_boolean(
        result,
        "flag_amostra_mercado",
    )
    result["flag_ipca_disponivel_qc"] = ensure_boolean(
        result,
        "flag_ipca_disponivel",
    )

    result["flag_setor_espacial_ausente"] = result["sector_join_key"].isna()

    lat = (
        pd.to_numeric(result["latitude"], errors="coerce")
        if "latitude" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    lon = (
        pd.to_numeric(result["longitude"], errors="coerce")
        if "longitude" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    result["flag_coordenada_ausente"] = lat.isna() | lon.isna()

    if "geocode_quality" in result.columns:
        quality = result["geocode_quality"].astype("string").str.lower()
        result["flag_geocode_utilizavel"] = quality.isin(["high", "medium"])
        result["flag_geocode_alta_confianca"] = quality.eq("high")
    elif "geocode_score_mean" in result.columns:
        score = pd.to_numeric(result["geocode_score_mean"], errors="coerce")
        result["flag_geocode_utilizavel"] = score.ge(80)
        result["flag_geocode_alta_confianca"] = score.ge(90)
    else:
        result["flag_geocode_utilizavel"] = ~result["flag_coordenada_ausente"]
        result["flag_geocode_alta_confianca"] = False

    current_year = datetime.now().year
    result["flag_ano_corrente_parcial"] = (
        pd.to_numeric(result["source_year"], errors="coerce").eq(current_year)
    )

    return result


def _safe_group_label(group_key: object) -> dict[str, Any]:
    """Processa a chave do groupby em formato serializável para auditoria."""
    if isinstance(group_key, tuple):
        return {
            f"group_{index + 1}": value
            for index, value in enumerate(group_key)
        }
    return {"group_1": group_key}


def add_log_iqr_outlier_flag(
    frame: pd.DataFrame,
    value_column: str,
    flag_column: str,
    metric_name: str,
    group_columns: list[str],
    multiplier: float,
    min_group_size: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Analisa outliers positivos em escala logarítmica com regra de Tukey."""
    result = frame.copy()
    result[flag_column] = False
    thresholds: list[dict[str, Any]] = []

    # Processa somente observações positivas porque a transformação log exige
    # domínio estritamente positivo; valores inválidos já têm flags próprias.
    values = pd.to_numeric(result[value_column], errors="coerce")
    valid = values.gt(0) & np.isfinite(values)

    working = result.loc[valid, group_columns].copy()
    working["_value"] = values.loc[valid]
    working["_row_index"] = working.index

    if working.empty:
        return result, thresholds

    for group_key, group in working.groupby(group_columns, dropna=False):
        positive = group["_value"].astype(float)
        n = len(positive)

        record: dict[str, Any] = {
            "metric": metric_name,
            "value_column": value_column,
            "iqr_multiplier": multiplier,
            "minimum_group_size": min_group_size,
            "n": n,
            "applied": False,
        }
        record.update(_safe_group_label(group_key))

        if n < min_group_size:
            record["reason_not_applied"] = "group_too_small"
            thresholds.append(record)
            continue

        log_values = np.log(positive)
        q1 = float(log_values.quantile(0.25))
        q3 = float(log_values.quantile(0.75))
        iqr = q3 - q1

        if not math.isfinite(iqr) or iqr <= 0:
            record["reason_not_applied"] = "non_positive_iqr"
            thresholds.append(record)
            continue

        lower_log = q1 - multiplier * iqr
        upper_log = q3 + multiplier * iqr
        lower = float(np.exp(lower_log))
        upper = float(np.exp(upper_log))

        outlier_mask = positive.lt(lower) | positive.gt(upper)
        outlier_indices = group.loc[outlier_mask, "_row_index"]

        result.loc[outlier_indices, flag_column] = True

        record.update(
            {
                "applied": True,
                "q1_log": q1,
                "q3_log": q3,
                "iqr_log": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outliers": int(outlier_mask.sum()),
            }
        )
        thresholds.append(record)

    return result, thresholds


def add_statistical_quality_flags(
    frame: pd.DataFrame,
    multiplier: float,
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analisa extremos de valor, valor por m² e área em grupos comparáveis."""
    result = frame.copy()

    group_columns = ["source_year"]
    if "flag_multiunit_qc" in result.columns:
        group_columns.append("flag_multiunit_qc")

    all_thresholds: list[dict[str, Any]] = []
    metrics = [
        (
            "base_de_calculo_real",
            "flag_outlier_base_real",
            "base_real",
        ),
        (
            "valor_m2_base_real",
            "flag_outlier_valor_m2_real",
            "valor_m2_real",
        ),
        (
            "area_referencia_m2",
            "flag_outlier_area",
            "area_referencia",
        ),
    ]

    for value_column, flag_column, metric_name in metrics:
        result, thresholds = add_log_iqr_outlier_flag(
            frame=result,
            value_column=value_column,
            flag_column=flag_column,
            metric_name=metric_name,
            group_columns=group_columns,
            multiplier=multiplier,
            min_group_size=min_group_size,
        )
        all_thresholds.extend(thresholds)

    result["flag_outlier_estatistico"] = (
        result["flag_outlier_base_real"]
        | result["flag_outlier_valor_m2_real"]
        | result["flag_outlier_area"]
    )

    return result, pd.DataFrame(all_thresholds)


def add_analysis_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Processa flags finais para mercado, espaço, socioeconomia e modelos."""
    result = frame.copy()

    structural_invalid = (
        result["flag_base_real_invalida"]
        | result["flag_area_invalida"]
        | result["flag_valor_m2_real_invalido"]
        | result["flag_percentual_transmitido_invalido"]
        | result["flag_cancelado_qc"]
        | result["flag_duplicado_qc"]
        | ~result["flag_ipca_disponivel_qc"]
    )

    # Processa a amostra de mercado a partir do critério criado na limpeza e
    # acrescenta os controles de qualidade desta etapa.
    result["flag_use_market_analysis"] = (
        result["flag_amostra_mercado_origem"]
        & ~structural_invalid
        & ~result["flag_outlier_estatistico"]
    )

    result["flag_use_spatial_analysis"] = (
        result["flag_use_market_analysis"]
        & result["flag_geocode_utilizavel"]
        & ~result["flag_setor_espacial_ausente"]
    )

    result["flag_use_socioeconomic_analysis"] = (
        result["flag_use_spatial_analysis"]
        & result["flag_census_matched"]
    )

    # Analisa uma amostra preliminar mais conservadora para modelos hedônicos:
    # restringe guias multiunidade, mas não define ainda a especificação final.
    result["flag_use_model_preliminar"] = (
        result["flag_use_socioeconomic_analysis"]
        & ~result["flag_multiunit_qc"]
    )

    return result


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Analisa cobertura, problemas e amostras finais por ano."""
    summary = (
        frame.groupby("source_year", dropna=False)
        .agg(
            transactions=("transaction_id", "size"),
            market_origin=("flag_amostra_mercado_origem", "sum"),
            census_matched=("flag_census_matched", "sum"),
            geocode_usable=("flag_geocode_utilizavel", "sum"),
            invalid_base_real=("flag_base_real_invalida", "sum"),
            invalid_area=("flag_area_invalida", "sum"),
            invalid_value_m2_real=("flag_valor_m2_real_invalido", "sum"),
            exact_duplicates=("flag_duplicado_qc", "sum"),
            multiunit=("flag_multiunit_qc", "sum"),
            outlier_base_real=("flag_outlier_base_real", "sum"),
            outlier_value_m2_real=("flag_outlier_valor_m2_real", "sum"),
            outlier_area=("flag_outlier_area", "sum"),
            outlier_any=("flag_outlier_estatistico", "sum"),
            use_market=("flag_use_market_analysis", "sum"),
            use_spatial=("flag_use_spatial_analysis", "sum"),
            use_socioeconomic=("flag_use_socioeconomic_analysis", "sum"),
            use_model_preliminary=("flag_use_model_preliminar", "sum"),
        )
        .reset_index()
        .sort_values("source_year")
    )

    denominator = summary["transactions"].replace(0, np.nan)
    for count_column in (
        "market_origin",
        "census_matched",
        "geocode_usable",
        "outlier_any",
        "use_market",
        "use_spatial",
        "use_socioeconomic",
        "use_model_preliminary",
    ):
        summary[f"pct_{count_column}"] = (
            100.0 * summary[count_column] / denominator
        ).round(3)

    return summary


def _count_true(frame: pd.DataFrame, column: str) -> int:
    """Processa a contagem segura de uma flag booleana."""
    if column not in frame.columns:
        return 0
    return int(frame[column].astype("boolean").fillna(False).sum())


def build_metadata(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    multiplier: float,
    min_group_size: int,
    output_path: Path,
    sample_path: Path,
) -> dict[str, Any]:
    """Analisa e documenta as decisões metodológicas do controle de qualidade."""
    current_year = datetime.now().year
    partial_year_rows = _count_true(frame, "flag_ano_corrente_parcial")

    return {
        "generated_at": utc_now_iso(),
        "methodology": {
            "outlier_method": "Tukey IQR em log(valores positivos)",
            "iqr_multiplier": multiplier,
            "minimum_group_size": min_group_size,
            "outlier_groups": [
                "source_year",
                "flag_multiunit_qc",
            ],
            "outlier_metrics": [
                "base_de_calculo_real",
                "valor_m2_base_real",
                "area_referencia_m2",
            ],
            "preserve_all_rows": True,
            "census_join": "sector_final_code == CD_SETOR",
        },
        "samples": {
            "market": (
                "amostra de mercado de origem + valores/área/IPCA válidos + "
                "sem duplicidade exata + sem outlier estatístico"
            ),
            "spatial": "amostra de mercado + geocodificação utilizável + setor",
            "socioeconomic": "amostra espacial + perfil censitário associado",
            "model_preliminary": "amostra socioeconômica + guia de uma unidade",
        },
        "quality": {
            "transactions": int(len(frame)),
            "census_matched": _count_true(frame, "flag_census_matched"),
            "outlier_any": _count_true(frame, "flag_outlier_estatistico"),
            "market_sample": _count_true(frame, "flag_use_market_analysis"),
            "spatial_sample": _count_true(frame, "flag_use_spatial_analysis"),
            "socioeconomic_sample": _count_true(
                frame,
                "flag_use_socioeconomic_analysis",
            ),
            "model_preliminary_sample": _count_true(
                frame,
                "flag_use_model_preliminar",
            ),
            "current_year": current_year,
            "current_year_partial_rows": partial_year_rows,
            "outlier_threshold_rows": int(len(thresholds)),
        },
        "notes": [
            (
                "Analisa a base_de_calculo como base tributária do ITBI e não "
                "como preço efetivo de venda."
            ),
            (
                "Processa outliers como flags; a base auditada preserva todas "
                "as observações."
            ),
            (
                "Analisa o ano corrente como parcial para evitar comparações "
                "anuais diretas sem ajuste de cobertura temporal."
            ),
            (
                "Processa a amostra preliminar de modelos de forma conservadora, "
                "excluindo guias multiunidade."
            ),
        ],
        "outputs": {
            "audited": str(output_path.relative_to(PROJECT_ROOT)),
            "analysis_sample": str(sample_path.relative_to(PROJECT_ROOT)),
        },
    }


def save_outputs(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    metadata: dict[str, Any],
    output_path: Path,
    sample_path: Path,
    summary_path: Path,
    thresholds_path: Path,
    metadata_path: Path,
) -> None:
    """Processa a persistência das bases e relatórios de controle de qualidade."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(output_path, index=False)

    # Processa uma amostra pronta para indicadores de mercado sem apagar a base
    # completa e auditável.
    sample = frame.loc[frame["flag_use_market_analysis"]].copy()
    sample.to_parquet(sample_path, index=False)

    summary.to_csv(summary_path, index=False, encoding="utf-8")
    thresholds.to_csv(thresholds_path, index=False, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Analisa qualidade e prepara a amostra analítica do ITBI."
    )
    parser.add_argument("--itbi", type=Path, default=DEFAULT_ITBI)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=DEFAULT_IQR_MULTIPLIER,
        help="Multiplicador do IQR em log. Padrão conservador: 3.0.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help="Tamanho mínimo do grupo para aplicar regra de outlier.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Analisa parâmetros que alteram as regras estatísticas."""
    if args.iqr_multiplier <= 0:
        raise ValueError("--iqr-multiplier deve ser maior que zero.")
    if args.min_group_size < 10:
        raise ValueError("--min-group-size deve ser pelo menos 10.")


def main() -> int:
    """Processa o fluxo completo de controle de qualidade."""
    args = parse_args()

    try:
        validate_args(args)

        itbi_path = resolve_path(args.itbi)
        census_path = resolve_path(args.census)
        output_path = resolve_path(args.output)
        sample_path = resolve_path(args.sample)
        summary_path = resolve_path(args.summary)
        thresholds_path = resolve_path(args.thresholds)
        metadata_path = resolve_path(args.metadata)

        itbi = load_itbi(itbi_path)
        census = load_census(census_path)

        enriched = enrich_with_census(itbi, census)
        enriched = add_structural_quality_flags(enriched)

        audited, thresholds = add_statistical_quality_flags(
            enriched,
            multiplier=args.iqr_multiplier,
            min_group_size=args.min_group_size,
        )
        audited = add_analysis_flags(audited)

        summary = build_summary(audited)
        metadata = build_metadata(
            frame=audited,
            thresholds=thresholds,
            multiplier=args.iqr_multiplier,
            min_group_size=args.min_group_size,
            output_path=output_path,
            sample_path=sample_path,
        )

        save_outputs(
            frame=audited,
            summary=summary,
            thresholds=thresholds,
            metadata=metadata,
            output_path=output_path,
            sample_path=sample_path,
            summary_path=summary_path,
            thresholds_path=thresholds_path,
            metadata_path=metadata_path,
        )

        total = len(audited)
        market = _count_true(audited, "flag_use_market_analysis")
        spatial = _count_true(audited, "flag_use_spatial_analysis")
        socioeconomic = _count_true(
            audited,
            "flag_use_socioeconomic_analysis",
        )
        outliers = _count_true(audited, "flag_outlier_estatistico")

        print("\n[ok] Controle de qualidade concluído")
        print(f"     transações auditadas: {total:,}")
        print(f"     outliers sinalizados: {outliers:,}")
        print(f"     amostra de mercado: {market:,}")
        print(f"     amostra espacial: {spatial:,}")
        print(f"     amostra socioeconômica: {socioeconomic:,}")
        print(f"     {output_path.relative_to(PROJECT_ROOT)}")
        print(f"     {sample_path.relative_to(PROJECT_ROOT)}")
        print(f"     {summary_path.relative_to(PROJECT_ROOT)}")
        print(f"     {thresholds_path.relative_to(PROJECT_ROOT)}")
        print(f"     {metadata_path.relative_to(PROJECT_ROOT)}")

        print("\nResumo por ano:")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(summary.to_string(index=False))

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
