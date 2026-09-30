#!/usr/bin/env python3
"""Processa a correção monetária das transações do ITBI pelo IPCA.

Entrada:
    data/processed/itbi_transactions_spatial.parquet
    data/raw/macro/ipca_mensal_sgs433_2020_ate_hoje.json

Saídas:
    data/processed/itbi_transactions_deflated.parquet
    data/processed/ipca_monthly_index.parquet
    data/processed/deflation_summary.csv
    data/processed/deflation_metadata.json

Metodologia:
- processa a data de pagamento como referência temporal principal;
- processa a data de estimativa como fallback quando o pagamento está ausente;
- analisa a disponibilidade mensal do IPCA antes de corrigir cada observação;
- processa um índice encadeado do IPCA mensal;
- converte valores nominais para reais na data-base selecionada;
- preserva todas as transações e sinaliza as que não podem ser corrigidas.

A correção usa:
    valor_real_ref = valor_nominal * (indice_ref / indice_mes_transacao)

Por padrão, a data-base é o último mês disponível na série baixada do IPCA.

Uso:
    python scripts/06_deflate_prices.py
    python scripts/06_deflate_prices.py --reference 2026-08

Dependências:
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

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_spatial.parquet"
)
DEFAULT_IPCA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "macro"
    / "ipca_mensal_sgs433_2020_ate_hoje.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_deflated.parquet"
)
DEFAULT_INDEX_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "ipca_monthly_index.parquet"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "deflation_summary.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "deflation_metadata.json"

DEFAULT_DATE_COLUMN = "data_pagamento"
DEFAULT_FALLBACK_DATE_COLUMN = "data_estimativa"

REQUIRED_ITBI_COLUMNS = {
    "transaction_id",
    "base_de_calculo",
    "valor_m2_base",
}


def utc_now_iso() -> str:
    """Processa o instante atual em UTC para metadados de reprodutibilidade."""
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    """Processa caminhos relativos sempre a partir da raiz do projeto."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_ipca(path: Path) -> pd.DataFrame:
    """Processa o JSON do BCB e constrói um índice mensal encadeado do IPCA."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo do IPCA não encontrado: {path}. "
            "Execute: python scripts/downloads.py --groups macro"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("O arquivo do IPCA não contém uma lista de observações.")

    frame = pd.DataFrame(payload)
    required = {"data", "valor"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"IPCA sem campos obrigatórios: {sorted(missing)}. "
            f"Campos encontrados: {list(frame.columns)}"
        )

    # Processa as datas no formato publicado pelo SGS e os percentuais mensais.
    frame["date"] = pd.to_datetime(
        frame["data"].astype("string").str.strip(),
        format="%d/%m/%Y",
        errors="coerce",
    )
    frame["ipca_monthly_pct"] = pd.to_numeric(
        frame["valor"]
        .astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # Analisa falhas de leitura antes de construir o índice para evitar correções
    # silenciosamente incorretas.
    invalid = frame["date"].isna() | frame["ipca_monthly_pct"].isna()
    if invalid.any():
        sample = frame.loc[invalid, ["data", "valor"]].head(5).to_dict("records")
        raise ValueError(
            f"IPCA contém {int(invalid.sum())} observações inválidas. "
            f"Exemplos: {sample}"
        )

    frame["period"] = frame["date"].dt.to_period("M")
    frame = frame.sort_values("period").reset_index(drop=True)

    # Analisa se cada mês aparece uma única vez, condição necessária para o
    # encadeamento mensal.
    duplicated = frame["period"].duplicated(keep=False)
    if duplicated.any():
        periods = sorted(frame.loc[duplicated, "period"].astype(str).unique())
        raise ValueError(f"IPCA contém meses duplicados: {periods[:10]}")

    # Processa o nível do índice em escala arbitrária; somente razões entre
    # níveis são usadas na correção monetária.
    frame["ipca_index_level"] = (
        1.0 + frame["ipca_monthly_pct"] / 100.0
    ).cumprod()

    return frame[
        ["date", "period", "ipca_monthly_pct", "ipca_index_level"]
    ].copy()


def resolve_reference_period(
    ipca: pd.DataFrame,
    reference: str,
) -> pd.Period:
    """Processa e valida o mês de referência escolhido para valores reais."""
    first_period = ipca["period"].min()
    last_period = ipca["period"].max()

    if reference.lower() == "latest":
        return last_period

    try:
        reference_period = pd.Period(reference, freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--reference deve ser 'latest' ou um mês no formato AAAA-MM."
        ) from exc

    # Analisa se a data-base está coberta pela série efetivamente baixada.
    if reference_period < first_period or reference_period > last_period:
        raise ValueError(
            f"Mês de referência {reference_period} fora da cobertura do IPCA "
            f"({first_period} a {last_period})."
        )

    if reference_period not in set(ipca["period"]):
        raise ValueError(
            f"Mês de referência {reference_period} não existe na série do IPCA."
        )

    return reference_period


def prepare_transaction_dates(
    transactions: pd.DataFrame,
    primary_column: str,
    fallback_column: str | None,
) -> pd.DataFrame:
    """Processa a data monetária de cada transação e registra sua origem."""
    if primary_column not in transactions.columns:
        raise ValueError(f"Coluna de data principal ausente: {primary_column}")

    result = transactions.copy()
    primary = pd.to_datetime(result[primary_column], errors="coerce")

    if fallback_column and fallback_column in result.columns:
        fallback = pd.to_datetime(result[fallback_column], errors="coerce")
    else:
        fallback = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")

    result["data_correcao_monetaria"] = primary.combine_first(fallback)
    result["fonte_data_correcao"] = "missing"
    result.loc[primary.notna(), "fonte_data_correcao"] = primary_column
    result.loc[
        primary.isna() & fallback.notna(),
        "fonte_data_correcao",
    ] = fallback_column or "fallback"

    result["flag_data_correcao_fallback"] = (
        primary.isna() & fallback.notna()
    )
    result["flag_data_correcao_ausente"] = result[
        "data_correcao_monetaria"
    ].isna()

    return result


def deflate_transactions(
    transactions: pd.DataFrame,
    ipca: pd.DataFrame,
    reference_period: pd.Period,
) -> pd.DataFrame:
    """Processa valores nominais para a moeda constante do mês de referência."""
    result = transactions.copy()
    transaction_period = result["data_correcao_monetaria"].dt.to_period("M")

    index_by_period = ipca.set_index("period")["ipca_index_level"]
    monthly_by_period = ipca.set_index("period")["ipca_monthly_pct"]

    reference_index = float(index_by_period.loc[reference_period])
    first_period = ipca["period"].min()
    last_period = ipca["period"].max()

    result["periodo_ipca_transacao"] = transaction_period.astype("string")
    result["periodo_ipca_referencia"] = str(reference_period)
    result["ipca_variacao_mes_transacao_pct"] = transaction_period.map(
        monthly_by_period
    )
    result["ipca_indice_transacao"] = transaction_period.map(index_by_period)
    result["ipca_indice_referencia"] = reference_index
    result["fator_correcao_ipca"] = (
        reference_index / result["ipca_indice_transacao"]
    )

    # Analisa datas fora da cobertura antes de calcular os valores reais.
    result["flag_data_antes_ipca"] = (
        transaction_period.notna() & (transaction_period < first_period)
    )
    result["flag_data_apos_ipca"] = (
        transaction_period.notna() & (transaction_period > last_period)
    )
    result["flag_ipca_disponivel"] = result["ipca_indice_transacao"].notna()

    base_nominal = pd.to_numeric(result["base_de_calculo"], errors="coerce")
    value_m2_nominal = pd.to_numeric(result["valor_m2_base"], errors="coerce")

    # Processa os valores reais sem substituir ou sobrescrever os valores
    # nominais existentes.
    result["base_de_calculo_real"] = (
        base_nominal * result["fator_correcao_ipca"]
    )
    result["valor_m2_base_real"] = (
        value_m2_nominal * result["fator_correcao_ipca"]
    )

    result["flag_base_real_calculada"] = (
        base_nominal.notna()
        & result["fator_correcao_ipca"].notna()
    )
    result["flag_valor_m2_real_calculado"] = (
        value_m2_nominal.notna()
        & result["fator_correcao_ipca"].notna()
    )

    return result


def build_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Analisa a cobertura e a distribuição nominal/real por ano."""
    frame = transactions.copy()

    if "source_year" not in frame.columns:
        frame["source_year"] = frame["data_correcao_monetaria"].dt.year.astype(
            "Int64"
        )

    if "flag_amostra_mercado" not in frame.columns:
        frame["flag_amostra_mercado"] = False

    summary = (
        frame.groupby("source_year", dropna=False)
        .agg(
            transactions=("transaction_id", "size"),
            market_sample_transactions=("flag_amostra_mercado", "sum"),
            correction_date_available=(
                "flag_data_correcao_ausente",
                lambda s: int((~s.fillna(True)).sum()),
            ),
            correction_date_fallback=("flag_data_correcao_fallback", "sum"),
            ipca_matched=("flag_ipca_disponivel", "sum"),
            base_real_calculated=("flag_base_real_calculada", "sum"),
            value_m2_real_calculated=("flag_valor_m2_real_calculado", "sum"),
            median_base_nominal=("base_de_calculo", "median"),
            median_base_real=("base_de_calculo_real", "median"),
            median_value_m2_nominal=("valor_m2_base", "median"),
            median_value_m2_real=("valor_m2_base_real", "median"),
        )
        .reset_index()
        .sort_values("source_year")
    )

    summary["pct_ipca_matched"] = (
        100.0 * summary["ipca_matched"] / summary["transactions"]
    ).round(3)

    return summary


def build_metadata(
    transactions: pd.DataFrame,
    ipca: pd.DataFrame,
    reference_period: pd.Period,
    primary_date: str,
    fallback_date: str | None,
    output_path: Path,
) -> dict[str, Any]:
    """Analisa e registra as premissas utilizadas na correção monetária."""
    total = len(transactions)

    return {
        "generated_at": utc_now_iso(),
        "source": {
            "series": "BCB SGS 433 - IPCA mensal",
            "input_period_start": str(ipca["period"].min()),
            "input_period_end": str(ipca["period"].max()),
        },
        "methodology": {
            "reference_period": str(reference_period),
            "primary_transaction_date": primary_date,
            "fallback_transaction_date": fallback_date,
            "formula": (
                "valor_real_ref = valor_nominal * "
                "(indice_ipca_ref / indice_ipca_mes_transacao)"
            ),
            "nominal_fields_preserved": [
                "base_de_calculo",
                "valor_m2_base",
            ],
            "real_fields_created": [
                "base_de_calculo_real",
                "valor_m2_base_real",
            ],
        },
        "quality": {
            "transactions": total,
            "correction_date_missing": int(
                transactions["flag_data_correcao_ausente"].sum()
            ),
            "correction_date_fallback": int(
                transactions["flag_data_correcao_fallback"].sum()
            ),
            "ipca_unmatched": int((~transactions["flag_ipca_disponivel"]).sum()),
            "dates_before_ipca": int(transactions["flag_data_antes_ipca"].sum()),
            "dates_after_ipca": int(transactions["flag_data_apos_ipca"].sum()),
            "base_real_calculated": int(
                transactions["flag_base_real_calculada"].sum()
            ),
            "value_m2_real_calculated": int(
                transactions["flag_valor_m2_real_calculado"].sum()
            ),
        },
        "notes": [
            (
                "Processa a base_de_calculo como base tributária do ITBI; "
                "não a reclassifica como preço efetivo de venda."
            ),
            (
                "Analisa transações sem IPCA disponível por flags e preserva "
                "essas linhas com valores reais ausentes."
            ),
            (
                "Processa a data de pagamento como padrão e utiliza a data de "
                "estimativa apenas quando configurada como fallback."
            ),
        ],
        "output": str(output_path.relative_to(PROJECT_ROOT)),
    }


def save_ipca_index(ipca: pd.DataFrame, path: Path) -> None:
    """Processa uma versão persistente do índice mensal usado na correção."""
    output = ipca.copy()
    output["period"] = output["period"].astype("string")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    """Processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Corrige monetariamente o ITBI de Porto Alegre pelo IPCA."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ipca", type=Path, default=DEFAULT_IPCA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index-output", type=Path, default=DEFAULT_INDEX_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--reference",
        default="latest",
        help="Mês de referência AAAA-MM. Padrão: último mês disponível.",
    )
    parser.add_argument(
        "--date-column",
        default=DEFAULT_DATE_COLUMN,
        help="Coluna principal para definir o mês da transação.",
    )
    parser.add_argument(
        "--fallback-date-column",
        default=DEFAULT_FALLBACK_DATE_COLUMN,
        help="Coluna usada quando a data principal estiver ausente.",
    )
    return parser.parse_args()


def main() -> int:
    """Processa o fluxo completo de correção monetária."""
    args = parse_args()

    input_path = resolve_path(args.input)
    ipca_path = resolve_path(args.ipca)
    output_path = resolve_path(args.output)
    index_output_path = resolve_path(args.index_output)
    summary_path = resolve_path(args.summary)
    metadata_path = resolve_path(args.metadata)

    if not input_path.exists():
        print(
            "[erro] Base espacial do ITBI não encontrada. Execute antes: "
            "python scripts/04_spatial_join.py",
            file=sys.stderr,
        )
        return 1

    try:
        transactions = pd.read_parquet(input_path)

        # Analisa o contrato mínimo da etapa anterior antes de processar valores.
        missing = REQUIRED_ITBI_COLUMNS - set(transactions.columns)
        if missing:
            raise ValueError(
                f"Base do ITBI sem colunas obrigatórias: {sorted(missing)}"
            )

        ipca = load_ipca(ipca_path)
        reference_period = resolve_reference_period(ipca, args.reference)

        transactions = prepare_transaction_dates(
            transactions,
            primary_column=args.date_column,
            fallback_column=args.fallback_date_column,
        )
        transactions = deflate_transactions(
            transactions,
            ipca,
            reference_period,
        )

        summary = build_summary(transactions)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        transactions.to_parquet(output_path, index=False)
        save_ipca_index(ipca, index_output_path)
        summary.to_csv(summary_path, index=False, encoding="utf-8")

        metadata = build_metadata(
            transactions=transactions,
            ipca=ipca,
            reference_period=reference_period,
            primary_date=args.date_column,
            fallback_date=args.fallback_date_column,
            output_path=output_path,
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        total = len(transactions)
        matched = int(transactions["flag_ipca_disponivel"].sum())
        base_real = int(transactions["flag_base_real_calculada"].sum())

        print("\n[ok] Correção monetária concluída")
        print(f"     data-base IPCA: {reference_period}")
        print(f"     transações: {total:,}")
        print(f"     com IPCA compatível: {matched:,}")
        print(f"     com base real calculada: {base_real:,}")
        print(f"     {output_path.relative_to(PROJECT_ROOT)}")
        print(f"     {index_output_path.relative_to(PROJECT_ROOT)}")
        print(f"     {summary_path.relative_to(PROJECT_ROOT)}")
        print(f"     {metadata_path.relative_to(PROJECT_ROOT)}")

        print("\nResumo por ano:")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(summary.to_string(index=False))

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
