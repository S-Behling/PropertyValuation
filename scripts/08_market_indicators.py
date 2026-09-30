#!/usr/bin/env python3
"""Processa indicadores do mercado imobiliário a partir da base auditada.

Entrada:
    data/processed/itbi_transactions_qc.parquet

Saídas:
    data/processed/market_indicators_city_year.parquet
    data/processed/market_indicators_neighborhood_year.parquet
    data/processed/market_indicators_planning_region_year.parquet
    data/processed/market_indicators_op_region_year.parquet
    data/processed/market_indicators_sector_year.parquet
    outputs/tables/market_indicators_city_year.csv
    outputs/tables/market_indicators_city_month.csv
    outputs/tables/market_indicators_city_ytd_comparable.csv
    outputs/tables/market_indicators_neighborhood_year.csv
    outputs/tables/market_indicators_planning_region_year.csv
    outputs/tables/market_indicators_op_region_year.csv
    outputs/tables/market_indicators_sector_year.csv
    data/processed/market_indicators_metadata.json

Metodologia:
- processa somente a amostra adequada a cada escala de análise;
- analisa a base_de_calculo como base tributária, não como preço efetivo de venda;
- processa valores monetários reais já corrigidos pelo IPCA;
- analisa anos incompletos antes de calcular variações anuais;
- processa uma tabela YTD comparável para o ano corrente;
- preserva medianas, quartis e contagens para reduzir dependência da média.

Uso:
    python scripts/08_market_indicators.py

Dependências:
    numpy
    pandas
    pyarrow
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_qc.parquet"
)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_METADATA = (
    PROJECT_ROOT / "data" / "processed" / "market_indicators_metadata.json"
)

REQUIRED_COLUMNS = {
    "transaction_id",
    "source_year",
    "base_de_calculo_real",
    "valor_m2_base_real",
    "area_referencia_m2",
    "flag_use_market_analysis",
    "flag_use_spatial_analysis",
}

METRIC_COLUMNS = [
    "n_transacoes",
    "n_unidades",
    "base_total_real",
    "base_media_real",
    "base_mediana_real",
    "base_p25_real",
    "base_p75_real",
    "valor_m2_medio_real",
    "valor_m2_mediano_real",
    "valor_m2_p25_real",
    "valor_m2_p75_real",
    "area_media_m2",
    "area_mediana_m2",
    "area_p25_m2",
    "area_p75_m2",
    "pct_multiunit",
]


def utc_now_iso() -> str:
    """Processa o instante atual em UTC para os metadados da execução."""
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    """Processa caminhos relativos sempre a partir da raiz do projeto."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_transactions(path: Path) -> pd.DataFrame:
    """Processa a base auditada e analisa seu contrato mínimo."""
    if not path.exists():
        raise FileNotFoundError(
            f"Base auditada não encontrada: {path}. "
            "Execute antes: python scripts/07_quality_control.py"
        )

    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Base auditada sem colunas obrigatórias: {sorted(missing)}"
        )

    if frame["transaction_id"].duplicated().any():
        count = int(frame["transaction_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"Base auditada contém {count} linhas com transaction_id duplicado."
        )

    return frame


def ensure_analysis_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Processa ano e mês analíticos usando a melhor data já consolidada."""
    result = frame.copy()

    if "data_correcao_monetaria" in result.columns:
        analysis_date = pd.to_datetime(
            result["data_correcao_monetaria"],
            errors="coerce",
        )
    else:
        payment = (
            pd.to_datetime(result["data_pagamento"], errors="coerce")
            if "data_pagamento" in result.columns
            else pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
        )
        estimate = (
            pd.to_datetime(result["data_estimativa"], errors="coerce")
            if "data_estimativa" in result.columns
            else pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
        )
        analysis_date = payment.combine_first(estimate)

    result["data_analise"] = analysis_date
    result["ano_analise"] = analysis_date.dt.year.astype("Int64")
    result["mes_analise"] = analysis_date.dt.month.astype("Int64")

    # Processa o ano do arquivo como fallback somente quando a data não existe.
    source_year = pd.to_numeric(result["source_year"], errors="coerce").astype("Int64")
    result["ano_analise"] = result["ano_analise"].combine_first(source_year)

    return result


def prepare_geographic_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Processa rótulos territoriais únicos sem apagar as fontes originais."""
    result = frame.copy()

    bairro_spatial = (
        result["bairro_spatial"].astype("string")
        if "bairro_spatial" in result.columns
        else pd.Series(pd.NA, index=result.index, dtype="string")
    )
    bairro_census = (
        result["census_bairro_nome"].astype("string")
        if "census_bairro_nome" in result.columns
        else pd.Series(pd.NA, index=result.index, dtype="string")
    )

    result["bairro_analise"] = bairro_spatial.combine_first(bairro_census)
    result["bairro_analise_fonte"] = "missing"
    result.loc[bairro_census.notna(), "bairro_analise_fonte"] = "censo"
    result.loc[bairro_spatial.notna(), "bairro_analise_fonte"] = "malha_pmpa"

    return result


def market_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Processa a amostra municipal válida para indicadores de mercado."""
    flag = frame["flag_use_market_analysis"].astype("boolean").fillna(False)
    return frame.loc[flag].copy()


def spatial_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Processa a amostra válida para indicadores territoriais."""
    flag = frame["flag_use_spatial_analysis"].astype("boolean").fillna(False)
    return frame.loc[flag].copy()


def quantile(series: pd.Series, q: float) -> float:
    """Processa um quantil numérico ignorando valores ausentes."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.quantile(q)) if len(numeric) else np.nan


def aggregate_market(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Processa indicadores robustos para um conjunto de agrupamentos."""
    if frame.empty:
        return pd.DataFrame(columns=group_columns + METRIC_COLUMNS)

    working = frame.copy()
    working["base_de_calculo_real"] = pd.to_numeric(
        working["base_de_calculo_real"],
        errors="coerce",
    )
    working["valor_m2_base_real"] = pd.to_numeric(
        working["valor_m2_base_real"],
        errors="coerce",
    )
    working["area_referencia_m2"] = pd.to_numeric(
        working["area_referencia_m2"],
        errors="coerce",
    )

    if "n_units" not in working.columns:
        working["n_units"] = 1

    if "flag_multiunit_qc" not in working.columns:
        working["flag_multiunit_qc"] = False

    grouped = working.groupby(group_columns, dropna=False, observed=True)

    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, keys, strict=False))

        base = group["base_de_calculo_real"]
        value_m2 = group["valor_m2_base_real"]
        area = group["area_referencia_m2"]
        n_units = pd.to_numeric(group["n_units"], errors="coerce").fillna(1)
        multiunit = group["flag_multiunit_qc"].astype("boolean").fillna(False)

        row.update(
            {
                "n_transacoes": int(len(group)),
                "n_unidades": int(n_units.sum()),
                "base_total_real": float(base.sum(min_count=1)),
                "base_media_real": float(base.mean()),
                "base_mediana_real": float(base.median()),
                "base_p25_real": quantile(base, 0.25),
                "base_p75_real": quantile(base, 0.75),
                "valor_m2_medio_real": float(value_m2.mean()),
                "valor_m2_mediano_real": float(value_m2.median()),
                "valor_m2_p25_real": quantile(value_m2, 0.25),
                "valor_m2_p75_real": quantile(value_m2, 0.75),
                "area_media_m2": float(area.mean()),
                "area_mediana_m2": float(area.median()),
                "area_p25_m2": quantile(area, 0.25),
                "area_p75_m2": quantile(area, 0.75),
                "pct_multiunit": float(100 * multiunit.mean()),
            }
        )

        if "sector_final_code" in group.columns:
            row["n_setores"] = int(
                group["sector_final_code"].astype("string").dropna().nunique()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def city_year_coverage(sample: pd.DataFrame) -> pd.DataFrame:
    """Analisa a cobertura mensal municipal para identificar anos incompletos."""
    valid = sample.loc[
        sample["ano_analise"].notna() & sample["mes_analise"].notna()
    ].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "ano_analise",
                "meses_observados",
                "primeiro_mes_observado",
                "ultimo_mes_observado",
                "flag_ano_incompleto",
            ]
        )

    coverage = (
        valid.groupby("ano_analise", observed=True)
        .agg(
            meses_observados=("mes_analise", "nunique"),
            primeiro_mes_observado=("mes_analise", "min"),
            ultimo_mes_observado=("mes_analise", "max"),
        )
        .reset_index()
    )

    current_year = datetime.now().year
    coverage["flag_ano_corrente"] = coverage["ano_analise"].eq(current_year)
    coverage["flag_ano_incompleto"] = (
        coverage["meses_observados"].lt(12)
        | coverage["flag_ano_corrente"]
    )

    return coverage


def add_city_year_growth(
    indicators: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Analisa variações anuais somente quando os dois anos são completos."""
    result = indicators.merge(
        coverage,
        on="ano_analise",
        how="left",
        validate="one_to_one",
    ).sort_values("ano_analise")

    previous_incomplete = result["flag_ano_incompleto"].shift(1).fillna(True)
    comparable = (
        ~result["flag_ano_incompleto"].fillna(True)
        & ~previous_incomplete.astype(bool)
    )

    growth_specs = {
        "n_transacoes": "variacao_transacoes_pct",
        "base_mediana_real": "variacao_base_mediana_real_pct",
        "valor_m2_mediano_real": "variacao_valor_m2_mediano_real_pct",
        "area_mediana_m2": "variacao_area_mediana_pct",
    }

    for source, target in growth_specs.items():
        raw_change = result[source].pct_change(fill_method=None) * 100
        result[target] = raw_change.where(comparable)

    return result


def add_geographic_market_share(
    indicators: pd.DataFrame,
    city_year: pd.DataFrame,
) -> pd.DataFrame:
    """Processa a participação de cada território no mercado municipal anual."""
    if indicators.empty:
        return indicators

    totals = city_year[
        ["ano_analise", "n_transacoes", "base_total_real"]
    ].rename(
        columns={
            "n_transacoes": "n_transacoes_cidade",
            "base_total_real": "base_total_real_cidade",
        }
    )

    result = indicators.merge(
        totals,
        on="ano_analise",
        how="left",
        validate="many_to_one",
    )

    result["participacao_transacoes_cidade_pct"] = (
        100
        * result["n_transacoes"]
        / result["n_transacoes_cidade"].replace(0, np.nan)
    )
    result["participacao_base_cidade_pct"] = (
        100
        * result["base_total_real"]
        / result["base_total_real_cidade"].replace(0, np.nan)
    )

    return result.drop(
        columns=["n_transacoes_cidade", "base_total_real_cidade"]
    )


def build_city_month(sample: pd.DataFrame) -> pd.DataFrame:
    """Processa indicadores mensais para analisar sazonalidade e cobertura."""
    valid = sample.loc[
        sample["ano_analise"].notna() & sample["mes_analise"].notna()
    ].copy()

    result = aggregate_market(
        valid,
        ["ano_analise", "mes_analise"],
    )

    if result.empty:
        return result

    result["periodo"] = (
        result["ano_analise"].astype("Int64").astype("string")
        + "-"
        + result["mes_analise"].astype("Int64").astype("string").str.zfill(2)
    )

    return result.sort_values(["ano_analise", "mes_analise"])


def build_ytd_comparable(
    sample: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Processa comparação YTD usando o mesmo mês-limite em todos os anos."""
    if sample.empty or coverage.empty:
        return pd.DataFrame()

    latest_year = int(coverage["ano_analise"].max())
    latest_row = coverage.loc[coverage["ano_analise"].eq(latest_year)].iloc[0]
    cutoff_month = int(latest_row["ultimo_mes_observado"])

    # Analisa somente anos que possuem observações até o mês-limite selecionado.
    eligible_years = coverage.loc[
        coverage["ultimo_mes_observado"].ge(cutoff_month),
        "ano_analise",
    ]

    ytd = sample.loc[
        sample["ano_analise"].isin(eligible_years)
        & sample["mes_analise"].between(1, cutoff_month, inclusive="both")
    ].copy()

    result = aggregate_market(ytd, ["ano_analise"])
    if result.empty:
        return result

    result["mes_limite_comparavel"] = cutoff_month
    result["periodo_comparavel"] = f"jan-{cutoff_month:02d}"

    result = result.sort_values("ano_analise")
    result["variacao_transacoes_ytd_pct"] = (
        result["n_transacoes"].pct_change(fill_method=None) * 100
    )
    result["variacao_valor_m2_mediano_ytd_pct"] = (
        result["valor_m2_mediano_real"].pct_change(fill_method=None) * 100
    )
    result["variacao_base_mediana_ytd_pct"] = (
        result["base_mediana_real"].pct_change(fill_method=None) * 100
    )

    return result


def build_geographic_table(
    sample: pd.DataFrame,
    geography_column: str,
    city_year: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Processa indicadores anuais de uma unidade territorial específica."""
    if geography_column not in sample.columns:
        return pd.DataFrame()

    valid = sample.loc[
        sample[geography_column].notna() & sample["ano_analise"].notna()
    ].copy()

    result = aggregate_market(
        valid,
        ["ano_analise", geography_column],
    )

    if result.empty:
        return result

    result = add_geographic_market_share(result, city_year)
    result = result.merge(
        coverage[
            [
                "ano_analise",
                "meses_observados",
                "ultimo_mes_observado",
                "flag_ano_incompleto",
            ]
        ],
        on="ano_analise",
        how="left",
        validate="many_to_one",
    )

    return result.sort_values(
        ["ano_analise", "n_transacoes"],
        ascending=[True, False],
    )


def save_table(
    frame: pd.DataFrame,
    parquet_path: Path | None,
    csv_path: Path,
) -> None:
    """Processa a persistência de uma tabela em formatos analíticos e legíveis."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8")

    if parquet_path is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet_path, index=False)


def build_metadata(
    input_frame: pd.DataFrame,
    market: pd.DataFrame,
    spatial: pd.DataFrame,
    city_year: pd.DataFrame,
    ytd: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Analisa a cobertura dos indicadores e documenta suas definições."""
    latest_year = (
        int(city_year["ano_analise"].max())
        if not city_year.empty
        else None
    )

    return {
        "generated_at": utc_now_iso(),
        "inputs": {
            "transactions_qc": int(len(input_frame)),
            "market_sample": int(len(market)),
            "spatial_sample": int(len(spatial)),
        },
        "methodology": {
            "monetary_basis": (
                "Valores reais já corrigidos pelo IPCA na etapa 06."
            ),
            "tax_basis_warning": (
                "base_de_calculo é a base tributária do ITBI e não é tratada "
                "como preço efetivo de venda."
            ),
            "main_price_indicator": "valor_m2_mediano_real",
            "annual_growth_rule": (
                "Variações anuais são calculadas apenas entre anos municipais "
                "completos; anos incompletos recebem valor ausente."
            ),
            "ytd_rule": (
                "A comparação YTD usa janeiro até o último mês observado do "
                "ano mais recente e aplica a mesma janela aos anos anteriores."
            ),
        },
        "latest_year": latest_year,
        "outputs": {
            name: {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
            }
            for name, frame in tables.items()
        },
        "notes": [
            (
                "Processa medianas e quartis como medidas principais por causa "
                "da assimetria típica de valores imobiliários."
            ),
            (
                "Analisa indicadores territoriais somente com a amostra espacial "
                "validada na etapa 07."
            ),
            (
                "Processa participação territorial com base no número de "
                "transações e na soma da base tributária real."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    """Processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Processa indicadores do mercado imobiliário de Porto Alegre."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def main() -> int:
    """Processa o fluxo completo de indicadores de mercado."""
    args = parse_args()

    try:
        input_path = resolve_path(args.input)
        processed_dir = resolve_path(args.processed_dir)
        tables_dir = resolve_path(args.tables_dir)
        metadata_path = resolve_path(args.metadata)

        transactions = load_transactions(input_path)
        transactions = ensure_analysis_date(transactions)
        transactions = prepare_geographic_labels(transactions)

        market = market_sample(transactions)
        spatial = spatial_sample(transactions)

        # Analisa cobertura temporal na escala municipal antes de comparar anos.
        coverage = city_year_coverage(market)

        city_year = aggregate_market(market, ["ano_analise"])
        city_year = add_city_year_growth(city_year, coverage)

        city_month = build_city_month(market)
        city_ytd = build_ytd_comparable(market, coverage)

        neighborhood_year = build_geographic_table(
            spatial,
            "bairro_analise",
            city_year,
            coverage,
        )
        planning_region_year = build_geographic_table(
            spatial,
            "regiao_planejamento",
            city_year,
            coverage,
        )
        op_region_year = build_geographic_table(
            spatial,
            "regiao_op",
            city_year,
            coverage,
        )
        sector_year = build_geographic_table(
            spatial,
            "sector_final_code",
            city_year,
            coverage,
        )

        outputs = {
            "city_year": city_year,
            "city_month": city_month,
            "city_ytd_comparable": city_ytd,
            "neighborhood_year": neighborhood_year,
            "planning_region_year": planning_region_year,
            "op_region_year": op_region_year,
            "sector_year": sector_year,
        }

        save_table(
            city_year,
            processed_dir / "market_indicators_city_year.parquet",
            tables_dir / "market_indicators_city_year.csv",
        )
        save_table(
            city_month,
            None,
            tables_dir / "market_indicators_city_month.csv",
        )
        save_table(
            city_ytd,
            None,
            tables_dir / "market_indicators_city_ytd_comparable.csv",
        )
        save_table(
            neighborhood_year,
            processed_dir / "market_indicators_neighborhood_year.parquet",
            tables_dir / "market_indicators_neighborhood_year.csv",
        )
        save_table(
            planning_region_year,
            processed_dir
            / "market_indicators_planning_region_year.parquet",
            tables_dir / "market_indicators_planning_region_year.csv",
        )
        save_table(
            op_region_year,
            processed_dir / "market_indicators_op_region_year.parquet",
            tables_dir / "market_indicators_op_region_year.csv",
        )
        save_table(
            sector_year,
            processed_dir / "market_indicators_sector_year.parquet",
            tables_dir / "market_indicators_sector_year.csv",
        )

        metadata = build_metadata(
            input_frame=transactions,
            market=market,
            spatial=spatial,
            city_year=city_year,
            ytd=city_ytd,
            tables=outputs,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print("\n[ok] Indicadores de mercado concluídos")
        print(f"     amostra de mercado: {len(market):,}")
        print(f"     amostra espacial: {len(spatial):,}")
        print(f"     anos municipais: {len(city_year):,}")
        print(f"     linhas bairro-ano: {len(neighborhood_year):,}")
        print(f"     linhas setor-ano: {len(sector_year):,}")
        print(f"     {metadata_path.relative_to(PROJECT_ROOT)}")

        print("\nIndicadores municipais por ano:")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(city_year.to_string(index=False))

        if not city_ytd.empty:
            print("\nComparação YTD:")
            with pd.option_context("display.max_columns", None, "display.width", 220):
                print(city_ytd.to_string(index=False))

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
