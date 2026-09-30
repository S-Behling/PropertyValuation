#!/usr/bin/env python3
"""Prepara e limpa os arquivos anuais de ITBI de Porto Alegre.

Entrada esperada:
    raw/porto_alegre/itbi/*.csv

Saídas:
    data/interim/itbi_units_clean.parquet
    data/interim/itbi_transactions_clean.parquet
    data/interim/itbi_year_summary.csv
    data/interim/itbi_column_profile.csv
    data/interim/itbi_quality_summary.json

Princípios:
- nunca altera os arquivos de raw/;
- preserva uma linha por unidade imobiliária publicada pela Prefeitura;
- não confunde "Base de Cálculo" do ITBI com preço efetivo de compra e venda;
- não distribui a Base de Cálculo entre unidades de uma guia multiunidade;
- não remove outliers nesta etapa: apenas cria flags de qualidade;
- infere grupos/guias multiunidade pela ordem original do arquivo porque o
  dicionário público informa que a Base de Cálculo aparece apenas na primeira
  unidade da guia e as demais vêm imediatamente abaixo sem valor nesse campo.

Uso:
    python scripts/02_prepare_itbi.py
    python scripts/02_prepare_itbi.py --input raw/porto_alegre/itbi
    python scripts/02_prepare_itbi.py --output data/interim

Dependências:
    pandas
    pyarrow
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "raw" / "porto_alegre" / "itbi"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim"

CANONICAL_COLUMNS = [
    "data_estimativa",
    "data_pagamento",
    "base_de_calculo",
    "perc_transmitido",
    "finalidade_construcao",
    "logradouro",
    "n_endereco",
    "n_unidade",
    "complemento_endereco",
    "bairro",
    "cep",
    "area_total_terreno",
    "area_constr_total",
    "area_constr_privativa",
    "ano_construcao",
    "n_matricula_reg_imoveis",
    "n_zona_reg_imoveis",
    "situacao",
]

REQUIRED_COLUMNS = {
    "data_estimativa",
    "data_pagamento",
    "base_de_calculo",
    "logradouro",
    "n_endereco",
    "bairro",
    "situacao",
}

NUMERIC_COLUMNS = [
    "base_de_calculo",
    "perc_transmitido",
    "area_total_terreno",
    "area_constr_total",
    "area_constr_privativa",
    "ano_construcao",
]

DATE_COLUMNS = ["data_estimativa", "data_pagamento"]

TEXT_COLUMNS = [
    "finalidade_construcao",
    "logradouro",
    "n_endereco",
    "n_unidade",
    "complemento_endereco",
    "bairro",
    "cep",
    "n_matricula_reg_imoveis",
    "n_zona_reg_imoveis",
    "situacao",
]

COLUMN_ALIASES = {
    # nomes oficiais do banco / cabeçalhos já normalizados
    "data_estimativa": "data_estimativa",
    "data_pagamento": "data_pagamento",
    "base_de_calculo": "base_de_calculo",
    "perc_transmitido": "perc_transmitido",
    "percentual_transmitido": "perc_transmitido",
    "finalidade_construcao": "finalidade_construcao",
    "logradouro": "logradouro",
    "n_endereco": "n_endereco",
    "numero_endereco": "n_endereco",
    "n_unidade": "n_unidade",
    "numero_unidade": "n_unidade",
    "complemento_endereco": "complemento_endereco",
    "bairro": "bairro",
    "cep": "cep",
    "area_total_terreno": "area_total_terreno",
    "area_constr_total": "area_constr_total",
    "area_construida_total": "area_constr_total",
    "area_constr_privativa": "area_constr_privativa",
    "area_construida_privativa": "area_constr_privativa",
    "ano_construcao": "ano_construcao",
    "n_matricula_reg_imoveis": "n_matricula_reg_imoveis",
    "numero_matricula_reg_imoveis": "n_matricula_reg_imoveis",
    "n_zona_reg_imoveis": "n_zona_reg_imoveis",
    "numero_zona_reg_imoveis": "n_zona_reg_imoveis",
    "situacao": "situacao",
}

NULL_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "<na>",
    "-",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_header(value: str) -> str:
    """Converte cabeçalhos variados para snake_case ASCII."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.strip().lower()
    text = re.sub(r"[º°]", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return pd.NA
    return re.sub(r"\s+", " ", text)


def normalize_key(value: object) -> object:
    """Texto para chaves de matching, sem acentos e em caixa alta."""
    value = normalize_text(value)
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip() or pd.NA


def normalize_integerish_text(value: object) -> object:
    """Mantém números de endereço/CEP como texto sem '.0' artificial."""
    value = normalize_text(value)
    if pd.isna(value):
        return pd.NA
    text = str(value)
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def parse_number_series(series: pd.Series) -> pd.Series:
    """Converte campo numérico; tolera formatação brasileira/americana."""
    s = series.astype("string").str.strip()
    s = s.replace(list(NULL_TOKENS), pd.NA)

    # Remove moeda e espaços; mantém dígitos, sinais e separadores.
    s = s.str.replace(r"[Rr]\$|\s", "", regex=True)
    s = s.str.replace(r"[^0-9,\.\-+]", "", regex=True)

    def convert_one(value: object) -> float | None:
        if pd.isna(value):
            return None
        text = str(value)
        if not text:
            return None

        if "," in text and "." in text:
            # O último separador é tratado como decimal.
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            # Vírgula única: decimal brasileiro.
            text = text.replace(",", ".")
        # Ponto único fica como decimal conforme dicionário oficial.

        try:
            return float(text)
        except ValueError:
            return None

    return pd.to_numeric(s.map(convert_one), errors="coerce")


def parse_date_series(series: pd.Series) -> pd.Series:
    """Datas do ITBI, priorizando dia/mês/ano."""
    s = series.astype("string").str.strip().replace(list(NULL_TOKENS), pd.NA)
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return parsed


def detect_csv_dialect(path: Path) -> tuple[str, str]:
    """Detecta separador; o dicionário oficial usa aspas simples em textos."""
    sample = path.read_bytes()[:65536].decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    # A documentação do ITBI informa aspas simples. Se não aparecem na amostra,
    # o quotechar continua inofensivo.
    quotechar = "'"
    return delimiter, quotechar


def read_itbi_csv(path: Path) -> pd.DataFrame:
    delimiter, quotechar = detect_csv_dialect(path)

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                sep=delimiter,
                quotechar=quotechar,
                encoding=encoding,
                dtype="string",
                keep_default_na=False,
                na_filter=False,
                engine="python",
                on_bad_lines="warn",
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def canonicalize_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    original_columns = list(df.columns)
    normalized = [normalize_header(col) for col in original_columns]

    rename: dict[str, str] = {}
    seen_targets: set[str] = set()

    for original, normal in zip(original_columns, normalized):
        target = COLUMN_ALIASES.get(normal, normal)
        if target in seen_targets:
            # Evita colisões silenciosas; mantém a coluna extra identificada.
            suffix = 2
            unique = f"{target}_{suffix}"
            while unique in seen_targets:
                suffix += 1
                unique = f"{target}_{suffix}"
            target = unique
        rename[original] = target
        seen_targets.add(target)

    df = df.rename(columns=rename)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{source.name}: faltam colunas obrigatórias: {sorted(missing)}. "
            f"Cabeçalhos encontrados: {list(df.columns)}"
        )

    # Garante schema estável mesmo se uma coluna opcional faltar em algum ano.
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    return df


def extract_year(path: Path, df: pd.DataFrame) -> int | None:
    match = re.search(r"(20\d{2})", path.name)
    if match:
        return int(match.group(1))

    if "data_estimativa" in df.columns:
        years = parse_date_series(df["data_estimativa"]).dt.year.dropna()
        if not years.empty:
            return int(years.mode().iloc[0])
    return None


def stable_hash(values: Iterable[object], prefix: str = "") -> str:
    parts = []
    for value in values:
        if pd.isna(value):
            parts.append("")
        else:
            parts.append(str(value))
    payload = "\x1f".join(parts)
    return prefix + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def add_source_metadata(df: pd.DataFrame, path: Path, year: int | None) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "source_file", path.name)
    df.insert(1, "source_year", year)
    df.insert(2, "source_row", range(2, len(df) + 2))  # linha 1 = cabeçalho
    return df


def clean_unit_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in TEXT_COLUMNS:
        df[column] = df[column].map(normalize_text).astype("string")

    df["n_endereco"] = df["n_endereco"].map(normalize_integerish_text).astype("string")
    df["cep"] = (
        df["cep"]
        .map(normalize_integerish_text)
        .astype("string")
        .str.replace(r"\D", "", regex=True)
        .str.zfill(8)
    )
    df.loc[df["cep"].str.len().ne(8).fillna(False), "cep"] = pd.NA

    for column in NUMERIC_COLUMNS:
        df[column] = parse_number_series(df[column])

    for column in DATE_COLUMNS:
        df[column] = parse_date_series(df[column])

    # Ano de construção é conceitualmente inteiro, mas nullable.
    df["ano_construcao"] = df["ano_construcao"].round().astype("Int64")

    # Chaves úteis para geocodificação/matching posterior.
    df["logradouro_key"] = df["logradouro"].map(normalize_key).astype("string")
    df["bairro_key"] = df["bairro"].map(normalize_key).astype("string")
    df["situacao_key"] = df["situacao"].map(normalize_key).astype("string")

    df["endereco_full"] = (
        df["logradouro"].fillna("")
        + ", "
        + df["n_endereco"].fillna("")
        + df["complemento_endereco"].fillna("").map(lambda x: f" - {x}" if x else "")
    ).str.replace(r"\s+", " ", regex=True).str.strip(" ,-")

    df["endereco_match_key"] = (
        df["logradouro_key"].fillna("")
        + "|"
        + df["n_endereco"].fillna("")
        + "|"
        + df["cep"].fillna("")
    )

    # Área por unidade. Mantemos todas as alternativas separadas.
    df["area_referencia_unit_m2"] = df["area_constr_privativa"]
    missing = df["area_referencia_unit_m2"].isna() | df["area_referencia_unit_m2"].le(0)
    df.loc[missing, "area_referencia_unit_m2"] = df.loc[missing, "area_constr_total"]
    missing = df["area_referencia_unit_m2"].isna() | df["area_referencia_unit_m2"].le(0)
    df.loc[missing, "area_referencia_unit_m2"] = df.loc[missing, "area_total_terreno"]

    # Flags básicas: não removemos registros aqui.
    df["flag_pago"] = df["data_pagamento"].notna()
    df["flag_cancelado"] = df["situacao_key"].eq("CANCELADO")
    df["flag_base_valida"] = df["base_de_calculo"].gt(0)
    df["flag_area_valida"] = df["area_referencia_unit_m2"].gt(0)
    df["flag_percentual_valido"] = (
        df["perc_transmitido"].isna()
        | (df["perc_transmitido"].gt(0) & df["perc_transmitido"].le(100))
    )

    current_year = datetime.now().year
    df["flag_ano_construcao_invalido"] = (
        df["ano_construcao"].notna()
        & (
            df["ano_construcao"].lt(1700)
            | df["ano_construcao"].gt(current_year + 1)
        )
    )

    # ID de conteúdo para identificar duplicidades exatas sem depender da linha.
    hash_columns = [
        "data_estimativa",
        "data_pagamento",
        "base_de_calculo",
        "perc_transmitido",
        "logradouro",
        "n_endereco",
        "n_unidade",
        "complemento_endereco",
        "bairro",
        "cep",
        "area_total_terreno",
        "area_constr_total",
        "area_constr_privativa",
        "ano_construcao",
        "n_matricula_reg_imoveis",
        "n_zona_reg_imoveis",
        "situacao",
    ]
    df["record_content_id"] = [
        stable_hash(row, prefix="unit_")
        for row in df[hash_columns].itertuples(index=False, name=None)
    ]
    df["flag_duplicado_exato"] = df.duplicated("record_content_id", keep=False)

    return df


def infer_guide_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Infere guias multiunidade preservando a ordem do CSV.

    Regra documentada pela Prefeitura:
    a primeira unidade da guia contém Base de Cálculo; unidades seguintes da
    mesma guia aparecem logo abaixo com Base de Cálculo vazia.
    """
    df = df.copy()

    groups = pd.Series(index=df.index, dtype="Int64")
    current_group = 0
    orphan_group = 0

    for idx, base in df["base_de_calculo"].items():
        if pd.notna(base):
            current_group += 1
            groups.loc[idx] = current_group
        elif current_group > 0:
            groups.loc[idx] = current_group
        else:
            # Linhas antes da primeira base não podem ser associadas com segurança.
            orphan_group += 1
            groups.loc[idx] = -orphan_group

    year = df["source_year"].iloc[0] if len(df) else None
    file_stub = normalize_header(Path(str(df["source_file"].iloc[0])).stem) if len(df) else "itbi"

    df["guide_seq_in_file"] = groups
    df["guide_group_id"] = groups.map(
        lambda n: f"guide_{year or 'na'}_{file_stub}_{'orphan_' if n < 0 else ''}{abs(int(n)):07d}"
    )
    df["flag_group_orphan"] = groups.lt(0)

    counts = df.groupby("guide_group_id")["guide_group_id"].transform("size")
    df["guide_n_units"] = counts.astype("Int64")
    df["flag_guide_multiunit"] = df["guide_n_units"].gt(1)

    # Base da guia é mantida como valor da guia, sem replicá-la como "preço"
    # de cada unidade. Preço/m² unitário só é calculado para guia de uma unidade.
    group_base = df.groupby("guide_group_id")["base_de_calculo"].transform("first")
    df["guide_base_de_calculo"] = group_base
    df["base_de_calculo_unitaria"] = group_base.where(df["guide_n_units"].eq(1))
    df["valor_m2_base_unitario"] = (
        df["base_de_calculo_unitaria"] / df["area_referencia_unit_m2"]
    )
    df.loc[
        ~df["flag_area_valida"] | ~df["flag_base_valida"],
        "valor_m2_base_unitario",
    ] = pd.NA

    df["flag_preco_unitario_confiavel"] = (
        df["guide_n_units"].eq(1)
        & df["flag_pago"]
        & ~df["flag_cancelado"]
        & df["flag_base_valida"]
        & df["flag_area_valida"]
    )

    return df


def first_non_null(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if len(values) else pd.NA


def unique_join(series: pd.Series) -> object:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.dropna():
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    return " | ".join(values) if values else pd.NA


def build_transaction_table(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for guide_id, group in units.groupby("guide_group_id", sort=False, dropna=False):
        group = group.sort_values("source_row")

        n_units = len(group)
        priv = group["area_constr_privativa"]
        total = group["area_constr_total"]
        terreno = group["area_total_terreno"]

        priv_valid = priv.gt(0)
        all_privative = bool(priv_valid.all()) if n_units else False

        if all_privative:
            area_ref = float(priv.sum())
            area_method = "sum_area_constr_privativa"
            area_complete = True
        elif n_units == 1 and pd.notna(total.iloc[0]) and total.iloc[0] > 0:
            area_ref = float(total.iloc[0])
            area_method = "area_constr_total_single_unit"
            area_complete = True
        elif n_units == 1 and pd.notna(terreno.iloc[0]) and terreno.iloc[0] > 0:
            area_ref = float(terreno.iloc[0])
            area_method = "area_total_terreno_single_unit"
            area_complete = True
        else:
            area_ref = None
            area_method = "indeterminada_multiunit"
            area_complete = False

        base = first_non_null(group["base_de_calculo"])
        valor_m2 = (
            float(base) / area_ref
            if pd.notna(base) and float(base) > 0 and area_ref and area_ref > 0
            else None
        )

        data_pagamento = first_non_null(group["data_pagamento"])
        situacoes = unique_join(group["situacao"])
        situacao_keys = set(group["situacao_key"].dropna().astype(str))

        row = {
            "transaction_id": guide_id,
            "source_file": first_non_null(group["source_file"]),
            "source_year": first_non_null(group["source_year"]),
            "source_row_start": int(group["source_row"].min()),
            "source_row_end": int(group["source_row"].max()),
            "n_units": n_units,
            "data_estimativa": first_non_null(group["data_estimativa"]),
            "data_pagamento": data_pagamento,
            "base_de_calculo": base,
            "perc_transmitido": first_non_null(group["perc_transmitido"]),
            "finalidade_construcao": unique_join(group["finalidade_construcao"]),
            "logradouro": unique_join(group["logradouro"]),
            "n_endereco": unique_join(group["n_endereco"]),
            "n_unidade": unique_join(group["n_unidade"]),
            "complemento_endereco": unique_join(group["complemento_endereco"]),
            "bairro": unique_join(group["bairro"]),
            "cep": unique_join(group["cep"]),
            "n_matricula_reg_imoveis": unique_join(group["n_matricula_reg_imoveis"]),
            "n_zona_reg_imoveis": unique_join(group["n_zona_reg_imoveis"]),
            "situacao": situacoes,
            "area_constr_privativa_soma_m2": float(priv[priv_valid].sum())
            if priv_valid.any()
            else None,
            "area_constr_privativa_n_validas": int(priv_valid.sum()),
            "area_constr_total_max_m2": float(total[total.gt(0)].max())
            if total.gt(0).any()
            else None,
            "area_total_terreno_max_m2": float(terreno[terreno.gt(0)].max())
            if terreno.gt(0).any()
            else None,
            "area_referencia_m2": area_ref,
            "area_referencia_metodo": area_method,
            "valor_m2_base": valor_m2,
            "flag_area_referencia_completa": area_complete,
            "flag_multiunit": n_units > 1,
            "flag_group_orphan": bool(group["flag_group_orphan"].any()),
            "flag_pago": pd.notna(data_pagamento),
            "flag_cancelado": "CANCELADO" in situacao_keys,
            "flag_duplicado_exato": bool(group["flag_duplicado_exato"].any()),
        }

        row["flag_amostra_mercado"] = bool(
            row["flag_pago"]
            and not row["flag_cancelado"]
            and pd.notna(base)
            and float(base) > 0
            and area_ref is not None
            and area_ref > 0
            and not row["flag_group_orphan"]
        )
        rows.append(row)

    tx = pd.DataFrame(rows)
    if tx.empty:
        return tx

    tx["source_year"] = tx["source_year"].astype("Int64")
    tx["n_units"] = tx["n_units"].astype("Int64")
    tx["area_constr_privativa_n_validas"] = tx[
        "area_constr_privativa_n_validas"
    ].astype("Int64")

    return tx


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)

    for column in df.columns:
        series = df[column]
        nulls = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        row = {
            "column": column,
            "dtype": str(series.dtype),
            "n_rows": total,
            "n_null": nulls,
            "pct_null": round(100 * nulls / total, 3) if total else None,
            "n_unique": unique,
        }

        if pd.api.types.is_numeric_dtype(series):
            valid = pd.to_numeric(series, errors="coerce").dropna()
            if len(valid):
                row.update(
                    {
                        "min": float(valid.min()),
                        "p25": float(valid.quantile(0.25)),
                        "median": float(valid.median()),
                        "mean": float(valid.mean()),
                        "p75": float(valid.quantile(0.75)),
                        "max": float(valid.max()),
                    }
                )

        rows.append(row)

    return pd.DataFrame(rows)


def year_summary(units: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    unit_summary = (
        units.groupby("source_year", dropna=False)
        .agg(
            unit_rows=("record_content_id", "size"),
            unique_unit_content=("record_content_id", "nunique"),
            exact_duplicate_rows=("flag_duplicado_exato", "sum"),
            paid_unit_rows=("flag_pago", "sum"),
            valid_base_unit_rows=("flag_base_valida", "sum"),
            valid_area_unit_rows=("flag_area_valida", "sum"),
            single_unit_price_rows=("flag_preco_unitario_confiavel", "sum"),
        )
        .reset_index()
    )

    tx_summary = (
        tx.groupby("source_year", dropna=False)
        .agg(
            transactions=("transaction_id", "size"),
            paid_transactions=("flag_pago", "sum"),
            market_sample_transactions=("flag_amostra_mercado", "sum"),
            multiunit_transactions=("flag_multiunit", "sum"),
            orphan_transactions=("flag_group_orphan", "sum"),
            median_base=("base_de_calculo", "median"),
            median_base_m2=("valor_m2_base", "median"),
        )
        .reset_index()
    )

    return unit_summary.merge(tx_summary, on="source_year", how="outer").sort_values(
        "source_year"
    )


def quality_summary(units: pd.DataFrame, tx: pd.DataFrame, files: list[Path]) -> dict:
    def count_true(frame: pd.DataFrame, column: str) -> int:
        return int(frame[column].fillna(False).sum()) if column in frame else 0

    return {
        "generated_at": utc_now_iso(),
        "input_files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "n_input_files": len(files),
        "units": {
            "rows": int(len(units)),
            "exact_duplicate_rows": count_true(units, "flag_duplicado_exato"),
            "paid_rows": count_true(units, "flag_pago"),
            "cancelled_rows": count_true(units, "flag_cancelado"),
            "invalid_construction_year_rows": count_true(
                units, "flag_ano_construcao_invalido"
            ),
            "multiunit_rows": count_true(units, "flag_guide_multiunit"),
            "orphan_group_rows": count_true(units, "flag_group_orphan"),
            "reliable_single_unit_price_rows": count_true(
                units, "flag_preco_unitario_confiavel"
            ),
        },
        "transactions": {
            "rows": int(len(tx)),
            "paid": count_true(tx, "flag_pago"),
            "cancelled": count_true(tx, "flag_cancelado"),
            "multiunit": count_true(tx, "flag_multiunit"),
            "orphan": count_true(tx, "flag_group_orphan"),
            "market_sample": count_true(tx, "flag_amostra_mercado"),
        },
        "notes": [
            "base_de_calculo é a base tributária publicada pela Prefeitura; não é renomeada como preço de venda.",
            "guias multiunidade são inferidas pela sequência de linhas conforme o dicionário oficial do ITBI.",
            "valor_m2_base_unitario só é produzido para guias com uma unidade.",
            "nenhum outlier estatístico é removido nesta etapa.",
        ],
    }


def prepare_itbi(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_files = sorted(
        path
        for path in input_dir.glob("*.csv")
        if path.is_file() and not path.name.startswith(".")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {input_dir}. "
            "Execute antes: python scripts/downloads.py --groups itbi"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for path in csv_files:
        print(f"[read] {path.relative_to(PROJECT_ROOT)}")
        raw = read_itbi_csv(path)
        raw = canonicalize_columns(raw, path)
        year = extract_year(path, raw)
        raw = add_source_metadata(raw, path, year)
        clean = clean_unit_records(raw)
        clean = infer_guide_groups(clean)
        frames.append(clean)

        print(
            f"       ano={year or 'NA'} | linhas={len(clean):,} | "
            f"pagas={int(clean['flag_pago'].sum()):,} | "
            f"multiunidade={int(clean['flag_guide_multiunit'].sum()):,}"
        )

    units = pd.concat(frames, ignore_index=True, sort=False)

    # ID global por ocorrência; preserva inclusive duplicatas exatas.
    units["unit_row_id"] = [
        stable_hash(
            [row.source_file, row.source_row, row.record_content_id],
            prefix="row_",
        )
        for row in units[["source_file", "source_row", "record_content_id"]].itertuples(
            index=False
        )
    ]

    # Reordena metadados no início.
    front = [
        "unit_row_id",
        "record_content_id",
        "source_file",
        "source_year",
        "source_row",
        "guide_group_id",
        "guide_seq_in_file",
        "guide_n_units",
    ]
    remaining = [column for column in units.columns if column not in front]
    units = units[front + remaining]

    transactions = build_transaction_table(units)

    units_path = output_dir / "itbi_units_clean.parquet"
    tx_path = output_dir / "itbi_transactions_clean.parquet"
    year_path = output_dir / "itbi_year_summary.csv"
    profile_path = output_dir / "itbi_column_profile.csv"
    quality_path = output_dir / "itbi_quality_summary.json"

    units.to_parquet(units_path, index=False)
    transactions.to_parquet(tx_path, index=False)

    summary = year_summary(units, transactions)
    summary.to_csv(year_path, index=False, encoding="utf-8")

    profile_columns(units).to_csv(profile_path, index=False, encoding="utf-8")

    quality_path.write_text(
        json.dumps(
            quality_summary(units, transactions, csv_files),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n[ok] Arquivos gerados:")
    for path in [units_path, tx_path, year_path, profile_path, quality_path]:
        print(f"     {path.relative_to(PROJECT_ROOT)}")

    print("\nResumo por ano:")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.to_string(index=False))

    return units, transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpa e padroniza os dados anuais de ITBI de Porto Alegre."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Diretório contendo os CSVs anuais de ITBI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Diretório para os arquivos intermediários.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input
    output_dir = args.output

    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    try:
        prepare_itbi(input_dir.resolve(), output_dir.resolve())
    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
