#!/usr/bin/env python3
"""Georreferencia unidades e transações do ITBI usando o CNEFE 2022/IBGE.

Entrada:
    data/interim/itbi_units_clean.parquet
    data/interim/itbi_transactions_clean.parquet
    data/raw/ibge/cnefe2022/4314902_PORTO_ALEGRE.zip

Saídas:
    data/interim/cnefe_address_index.parquet
    data/processed/itbi_units_geocoded.parquet
    data/processed/itbi_transactions_geocoded.parquet
    data/processed/geocoding_summary.csv
    data/processed/geocoding_quality.json

Estratégia de matching, da maior para a menor confiança:
1. logradouro + número + CEP exatos;
2. logradouro + número exatos, quando o endereço CNEFE é espacialmente consistente;
3. número + CEP e logradouro fuzzy;
4. número exato e logradouro fuzzy em toda a cidade;
5. opcionalmente, centroide dos pontos CNEFE do logradouro + CEP.

O CNEFE é usado como cadastro de referência. O script preserva COD_SETOR,
LATITUDE, LONGITUDE e NV_GEO_COORD do cadastro e registra método e score do
matching, permitindo filtrar geocodificações de menor qualidade depois.

Uso:
    python scripts/03_geocode_itbi.py
    python scripts/03_geocode_itbi.py --rebuild-index
    python scripts/03_geocode_itbi.py --allow-street-centroid

Dependências:
    pandas
    pyarrow
    rapidfuzz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNITS = PROJECT_ROOT / "data" / "interim" / "itbi_units_clean.parquet"
DEFAULT_TRANSACTIONS = (
    PROJECT_ROOT / "data" / "interim" / "itbi_transactions_clean.parquet"
)
DEFAULT_CNEFE_DIR = PROJECT_ROOT / "data" / "raw" / "ibge" / "cnefe2022"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "interim" / "cnefe_address_index.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"

PORTO_ALEGRE_IBGE = "4314902"

STREET_TYPE_TOKENS = {
    "R",
    "RUA",
    "AV",
    "AVENIDA",
    "AVEN",
    "TV",
    "TRAVESSA",
    "AL",
    "ALAMEDA",
    "EST",
    "ESTRADA",
    "ROD",
    "RODOVIA",
    "BC",
    "BECO",
    "PRACA",
    "PCA",
    "LARGO",
    "VIA",
    "ACESSO",
    "PASSAGEM",
}

CNEFE_WANTED = {
    "COD_UNICO_ENDERECO",
    "COD_MUNICIPIO",
    "COD_SETOR",
    "CEP",
    "DSC_LOCALIDADE",
    "NOM_TIPO_SEGLOGR",
    "NOM_TITULO_SEGLOGR",
    "NOM_SEGLOGR",
    "NUM_ENDERECO",
    "DSC_MODIFICADOR",
    "LATITUDE",
    "LONGITUDE",
    "NV_GEO_COORD",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_header(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return re.sub(r"\s+", " ", text)


def ascii_upper(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.upper()


def normalize_street(value: object) -> str | None:
    text = ascii_upper(value)
    if text is None:
        return None

    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    tokens = [token for token in re.split(r"\s+", text) if token]

    while tokens and tokens[0] in STREET_TYPE_TOKENS:
        tokens.pop(0)

    # Remover pontuação já torna "AV.", "R." etc. comparáveis; mantemos títulos
    # (DOUTOR, PROFESSOR, GENERAL...) porque fazem parte do nome do logradouro.
    result = " ".join(tokens)
    return result or None


def normalize_number(value: object) -> str | None:
    text = ascii_upper(value)
    if text is None or text in {"SN", "S N", "S/N", "SEM NUMERO"}:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return str(int(match.group(0)))


def normalize_cep(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if len(digits) < 8:
        digits = digits.zfill(8)
    return digits[:8] if len(digits) >= 8 else None


def normalize_sector(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    return digits[:15] if len(digits) >= 15 else (digits or None)


def safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", ".")
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def mode_or_none(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return None
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def present(value: object) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def group_spread_m(group: pd.DataFrame) -> float:
    lat_min = group["latitude"].min()
    lat_max = group["latitude"].max()
    lon_min = group["longitude"].min()
    lon_max = group["longitude"].max()
    if any(pd.isna(v) for v in (lat_min, lat_max, lon_min, lon_max)):
        return float("nan")
    return haversine_m(float(lat_min), float(lon_min), float(lat_max), float(lon_max))


def find_cnefe_zip(cnefe_dir: Path) -> Path:
    preferred = cnefe_dir / "4314902_PORTO_ALEGRE.zip"
    if preferred.exists():
        return preferred
    candidates = sorted(cnefe_dir.glob("*PORTO*ALEGRE*.zip"))
    if not candidates:
        candidates = sorted(cnefe_dir.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum ZIP do CNEFE encontrado em {cnefe_dir}. "
            "Execute: python scripts/downloads.py --groups cnefe"
        )
    return candidates[0]


def detect_delimiter(sample: bytes) -> str:
    text = sample.decode("utf-8-sig", errors="replace")
    try:
        return csv.Sniffer().sniff(text, delimiters=";,\t|").delimiter
    except csv.Error:
        return ";"


def read_cnefe(zip_path: Path) -> pd.DataFrame:
    print(f"[cnefe] lendo {zip_path.relative_to(PROJECT_ROOT)}")

    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.endswith("/")
        ]
        if not csv_members:
            raise ValueError(f"{zip_path.name}: nenhum CSV encontrado no ZIP")

        member = sorted(csv_members, key=lambda x: (len(x), x))[0]
        with archive.open(member) as stream:
            delimiter = detect_delimiter(stream.read(65536))

        with archive.open(member) as stream:
            header = pd.read_csv(
                stream,
                sep=delimiter,
                dtype="string",
                nrows=0,
                encoding="utf-8-sig",
                engine="python",
            )

        normalized = {column: normalize_header(column) for column in header.columns}
        available = set(normalized.values())
        required = {
            "COD_SETOR",
            "CEP",
            "NOM_SEGLOGR",
            "NUM_ENDERECO",
            "LATITUDE",
            "LONGITUDE",
        }
        missing = required - available
        if missing:
            raise ValueError(
                f"CNEFE sem campos obrigatórios {sorted(missing)}. "
                f"Campos encontrados: {sorted(available)}"
            )

        usecols = [
            original
            for original, canonical in normalized.items()
            if canonical in CNEFE_WANTED
        ]

        with archive.open(member) as stream:
            try:
                frame = pd.read_csv(
                    stream,
                    sep=delimiter,
                    usecols=usecols,
                    dtype="string",
                    encoding="utf-8-sig",
                    engine="python",
                )
            except UnicodeDecodeError:
                stream.close()
                with archive.open(member) as stream2:
                    frame = pd.read_csv(
                        stream2,
                        sep=delimiter,
                        usecols=usecols,
                        dtype="string",
                        encoding="latin-1",
                        engine="python",
                    )

    frame = frame.rename(columns={col: normalize_header(col) for col in frame.columns})

    if "COD_MUNICIPIO" in frame:
        municipality = (
            frame["COD_MUNICIPIO"].astype("string").str.replace(r"\D", "", regex=True)
        )
        frame = frame.loc[municipality.eq(PORTO_ALEGRE_IBGE)].copy()

    return frame


def build_cnefe_index(cnefe: pd.DataFrame) -> pd.DataFrame:
    c = cnefe.copy()

    for column in (
        "NOM_TIPO_SEGLOGR",
        "NOM_TITULO_SEGLOGR",
        "NOM_SEGLOGR",
        "NUM_ENDERECO",
        "CEP",
        "COD_SETOR",
        "NV_GEO_COORD",
    ):
        if column not in c:
            c[column] = pd.NA

    street_full = (
        c["NOM_TIPO_SEGLOGR"].fillna("")
        + " "
        + c["NOM_TITULO_SEGLOGR"].fillna("")
        + " "
        + c["NOM_SEGLOGR"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    c["street_label"] = street_full
    c["street_key"] = street_full.map(normalize_street)
    c["number_key"] = c["NUM_ENDERECO"].map(normalize_number)
    c["cep_key"] = c["CEP"].map(normalize_cep)
    c["sector_code"] = c["COD_SETOR"].map(normalize_sector)
    c["latitude"] = c["LATITUDE"].map(safe_float)
    c["longitude"] = c["LONGITUDE"].map(safe_float)

    valid_coords = (
        c["latitude"].between(-90, 90, inclusive="both")
        & c["longitude"].between(-180, 180, inclusive="both")
    )
    c = c.loc[
        valid_coords & c["street_key"].notna() & c["number_key"].notna()
    ].copy()

    if c.empty:
        raise ValueError("Nenhum endereço CNEFE válido com coordenadas.")

    rows: list[dict[str, Any]] = []
    group_cols = ["street_key", "number_key", "cep_key"]

    for keys, group in c.groupby(group_cols, dropna=False, sort=False):
        street_key, number_key, cep_key = keys
        rows.append(
            {
                "street_key": street_key,
                "number_key": number_key,
                "cep_key": cep_key,
                "cnefe_street_label": mode_or_none(group["street_label"]),
                "latitude": float(group["latitude"].median()),
                "longitude": float(group["longitude"].median()),
                "cnefe_sector_code": mode_or_none(group["sector_code"]),
                "cnefe_n_sector_codes": int(group["sector_code"].dropna().nunique()),
                "cnefe_geo_level": mode_or_none(group["NV_GEO_COORD"]),
                "cnefe_n_records": int(len(group)),
                "cnefe_spread_m": group_spread_m(group),
            }
        )

    index = pd.DataFrame(rows)
    index["cep_key"] = index["cep_key"].astype("string")
    index["number_key"] = index["number_key"].astype("string")
    index["street_key"] = index["street_key"].astype("string")
    index["cnefe_sector_code"] = index["cnefe_sector_code"].astype("string")
    return index


def load_or_build_index(
    cnefe_dir: Path,
    index_path: Path,
    rebuild: bool,
) -> pd.DataFrame:
    if index_path.exists() and not rebuild:
        print(f"[cnefe] usando índice {index_path.relative_to(PROJECT_ROOT)}")
        return pd.read_parquet(index_path)

    zip_path = find_cnefe_zip(cnefe_dir)
    cnefe = read_cnefe(zip_path)
    index = build_cnefe_index(cnefe)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(index_path, index=False)
    print(
        f"[cnefe] índice criado: {len(index):,} endereços -> "
        f"{index_path.relative_to(PROJECT_ROOT)}"
    )
    return index


def make_lookup(index: pd.DataFrame) -> dict[str, Any]:
    exact3: dict[tuple[str, str, str], dict] = {}
    exact2_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    cep_num: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_number: dict[str, list[dict]] = defaultdict(list)
    street_cep: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for record in index.to_dict("records"):
        street = record.get("street_key")
        number = record.get("number_key")
        cep = record.get("cep_key")
        if not present(street) or not present(number):
            continue

        if present(cep):
            exact3[(str(street), str(number), str(cep))] = record
            cep_num[(str(cep), str(number))].append(record)
            street_cep[(str(street), str(cep))].append(record)

        exact2_groups[(str(street), str(number))].append(record)
        by_number[str(number)].append(record)

    exact2: dict[tuple[str, str], dict] = {}
    for key, records in exact2_groups.items():
        if len(records) == 1:
            exact2[key] = records[0]
            continue

        temp = pd.DataFrame(records)
        spread = group_spread_m(temp)
        sectors = temp["cnefe_sector_code"].dropna().nunique()
        if (not math.isnan(spread) and spread <= 150) and sectors <= 1:
            combined = dict(records[0])
            combined["latitude"] = float(temp["latitude"].median())
            combined["longitude"] = float(temp["longitude"].median())
            combined["cnefe_n_records"] = int(temp["cnefe_n_records"].sum())
            combined["cnefe_spread_m"] = float(spread)
            exact2[key] = combined

    return {
        "exact3": exact3,
        "exact2": exact2,
        "cep_num": cep_num,
        "by_number": by_number,
        "street_cep": street_cep,
    }


def choose_fuzzy(
    street: str,
    candidates: list[dict],
    threshold: float,
    margin: float = 4.0,
) -> tuple[dict | None, float | None]:
    scored: list[tuple[float, dict]] = []
    for candidate in candidates:
        candidate_street = candidate.get("street_key")
        if not candidate_street:
            continue
        score = float(fuzz.ratio(street, str(candidate_street)))
        scored.append((score, candidate))

    if not scored:
        return None, None

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1

    if best_score < threshold:
        return None, best_score
    if len(scored) > 1 and (best_score - second) < margin:
        return None, best_score
    return best, best_score


def centroid_record(records: list[dict]) -> dict | None:
    if not records:
        return None
    frame = pd.DataFrame(records)
    sectors = frame["cnefe_sector_code"].dropna()
    return {
        "street_key": mode_or_none(frame["street_key"]),
        "number_key": None,
        "cep_key": mode_or_none(frame["cep_key"]),
        "cnefe_street_label": mode_or_none(frame["cnefe_street_label"]),
        "latitude": float(frame["latitude"].median()),
        "longitude": float(frame["longitude"].median()),
        "cnefe_sector_code": mode_or_none(sectors),
        "cnefe_n_sector_codes": int(sectors.nunique()),
        "cnefe_geo_level": mode_or_none(frame["cnefe_geo_level"]),
        "cnefe_n_records": int(frame["cnefe_n_records"].sum()),
        "cnefe_spread_m": group_spread_m(frame),
    }


def match_one_address(
    street: str | None,
    number: str | None,
    cep: str | None,
    lookup: dict[str, Any],
    allow_street_centroid: bool,
) -> dict[str, Any]:
    result = {
        "latitude": None,
        "longitude": None,
        "cnefe_sector_code": None,
        "cnefe_geo_level": None,
        "cnefe_n_records": None,
        "cnefe_spread_m": None,
        "cnefe_street_label": None,
        "geocode_method": "unmatched",
        "geocode_score": 0.0,
    }

    if not present(street):
        return result

    record: dict | None = None
    method = "unmatched"
    score = 0.0

    if present(number) and present(cep):
        record = lookup["exact3"].get((str(street), str(number), str(cep)))
        if record is not None:
            method, score = "exact_street_number_cep", 100.0

    if record is None and present(number):
        record = lookup["exact2"].get((str(street), str(number)))
        if record is not None:
            method, score = "exact_street_number", 96.0

    if record is None and present(number) and present(cep):
        candidates = lookup["cep_num"].get((str(cep), str(number)), [])
        if candidates:
            record, fuzzy_score = choose_fuzzy(street, candidates, threshold=78.0)
            if record is not None and fuzzy_score is not None:
                method = "fuzzy_street_exact_number_cep"
                score = min(94.0, 75.0 + fuzzy_score * 0.20)

    if record is None and present(number):
        candidates = lookup["by_number"].get(str(number), [])
        # Evita fuzzy global muito ambíguo em números extremamente comuns.
        if 0 < len(candidates) <= 500:
            record, fuzzy_score = choose_fuzzy(
                street, candidates, threshold=92.0, margin=5.0
            )
            if record is not None and fuzzy_score is not None:
                method = "fuzzy_street_exact_number"
                score = min(92.0, 70.0 + fuzzy_score * 0.22)

    if record is None and allow_street_centroid and present(cep):
        candidates = lookup["street_cep"].get((str(street), str(cep)), [])
        if candidates:
            record = centroid_record(candidates)
            if record is not None:
                method, score = "street_cep_centroid", 70.0

    if record is None:
        return result

    result.update(
        {
            "latitude": record.get("latitude"),
            "longitude": record.get("longitude"),
            "cnefe_sector_code": record.get("cnefe_sector_code"),
            "cnefe_geo_level": record.get("cnefe_geo_level"),
            "cnefe_n_records": record.get("cnefe_n_records"),
            "cnefe_spread_m": record.get("cnefe_spread_m"),
            "cnefe_street_label": record.get("cnefe_street_label"),
            "geocode_method": method,
            "geocode_score": round(float(score), 2),
        }
    )
    return result


def prepare_unique_itbi_addresses(units: pd.DataFrame) -> pd.DataFrame:
    required = {"logradouro", "n_endereco", "cep"}
    missing = required - set(units.columns)
    if missing:
        raise ValueError(
            f"itbi_units_clean.parquet sem campos necessários: {sorted(missing)}"
        )

    addresses = units[["logradouro", "n_endereco", "cep"]].copy()
    addresses["street_key_geo"] = addresses["logradouro"].map(normalize_street)
    addresses["number_key_geo"] = addresses["n_endereco"].map(normalize_number)
    addresses["cep_key_geo"] = addresses["cep"].map(normalize_cep)

    return addresses[
        ["street_key_geo", "number_key_geo", "cep_key_geo"]
    ].drop_duplicates(ignore_index=True)


def geocode_units(
    units: pd.DataFrame,
    cnefe_index: pd.DataFrame,
    allow_street_centroid: bool,
) -> pd.DataFrame:
    lookup = make_lookup(cnefe_index)
    addresses = prepare_unique_itbi_addresses(units)

    print(f"[match] {len(addresses):,} endereços únicos do ITBI")
    matches: list[dict[str, Any]] = []

    for i, row in enumerate(addresses.itertuples(index=False), start=1):
        matched = match_one_address(
            street=row.street_key_geo,
            number=row.number_key_geo,
            cep=row.cep_key_geo,
            lookup=lookup,
            allow_street_centroid=allow_street_centroid,
        )
        matched.update(
            {
                "street_key_geo": row.street_key_geo,
                "number_key_geo": row.number_key_geo,
                "cep_key_geo": row.cep_key_geo,
            }
        )
        matches.append(matched)
        if i % 25000 == 0:
            print(f"        processados {i:,}/{len(addresses):,}")

    match_df = pd.DataFrame(matches)
    addresses = addresses.merge(
        match_df,
        on=["street_key_geo", "number_key_geo", "cep_key_geo"],
        how="left",
        validate="one_to_one",
    )

    result = units.copy()
    result["street_key_geo"] = result["logradouro"].map(normalize_street)
    result["number_key_geo"] = result["n_endereco"].map(normalize_number)
    result["cep_key_geo"] = result["cep"].map(normalize_cep)

    result = result.merge(
        addresses,
        on=["street_key_geo", "number_key_geo", "cep_key_geo"],
        how="left",
        validate="many_to_one",
    )

    result["flag_geocoded"] = result["latitude"].notna() & result["longitude"].notna()
    result["flag_geocode_high_confidence"] = (
        result["flag_geocoded"] & result["geocode_score"].ge(90)
    )
    result["flag_geocode_medium_confidence"] = (
        result["flag_geocoded"]
        & result["geocode_score"].ge(80)
        & result["geocode_score"].lt(90)
    )
    result["flag_geocode_low_confidence"] = (
        result["flag_geocoded"] & result["geocode_score"].lt(80)
    )

    return result


def aggregate_transaction_geocodes(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for guide_id, group in units.groupby("guide_group_id", sort=False):
        good = group.loc[group["flag_geocoded"]].copy()
        n_total = len(group)
        n_geo = len(good)

        if n_geo:
            sectors = good["cnefe_sector_code"].dropna().astype("string")
            unique_sectors = int(sectors.nunique())
            sector = sectors.iloc[0] if unique_sectors == 1 else pd.NA

            address_keys = (
                good["street_key_geo"].fillna("")
                + "|"
                + good["number_key_geo"].fillna("")
                + "|"
                + good["cep_key_geo"].fillna("")
            )

            lat = float(good["latitude"].median())
            lon = float(good["longitude"].median())
            score_min = float(good["geocode_score"].min())
            score_mean = float(good["geocode_score"].mean())
            methods = " | ".join(
                sorted(set(good["geocode_method"].dropna().astype(str)))
            )

            if n_geo == n_total and score_min >= 90 and unique_sectors <= 1:
                quality = "high"
            elif n_geo / n_total >= 0.5 and score_mean >= 80:
                quality = "medium"
            else:
                quality = "low"
        else:
            unique_sectors = 0
            sector = pd.NA
            address_keys = pd.Series(dtype="string")
            lat = None
            lon = None
            score_min = None
            score_mean = None
            methods = "unmatched"
            quality = "unmatched"

        rows.append(
            {
                "transaction_id": guide_id,
                "latitude": lat,
                "longitude": lon,
                "cnefe_sector_code": sector,
                "geocode_quality": quality,
                "geocode_methods": methods,
                "geocode_score_min": score_min,
                "geocode_score_mean": score_mean,
                "n_units_total": n_total,
                "n_units_geocoded": n_geo,
                "pct_units_geocoded": 100.0 * n_geo / n_total if n_total else None,
                "n_unique_geocoded_addresses": int(address_keys.nunique())
                if n_geo
                else 0,
                "n_unique_cnefe_sectors": unique_sectors,
                "flag_multiple_cnefe_sectors": unique_sectors > 1,
            }
        )

    return pd.DataFrame(rows)


def build_summary(units: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        units.groupby(["source_year", "geocode_method"], dropna=False)
        .agg(
            rows=("unit_row_id", "size"),
            geocoded=("flag_geocoded", "sum"),
            mean_score=("geocode_score", "mean"),
        )
        .reset_index()
    )

    totals = (
        units.groupby("source_year", dropna=False)
        .agg(total_rows=("unit_row_id", "size"), total_geocoded=("flag_geocoded", "sum"))
        .reset_index()
    )
    totals["pct_geocoded"] = (
        100.0 * totals["total_geocoded"] / totals["total_rows"]
    )

    return grouped.merge(totals, on="source_year", how="left").sort_values(
        ["source_year", "rows"], ascending=[True, False]
    )


def quality_payload(units: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    methods = (
        units["geocode_method"].fillna("unmatched").value_counts(dropna=False).to_dict()
    )
    return {
        "generated_at": utc_now_iso(),
        "units": {
            "rows": int(len(units)),
            "geocoded": int(units["flag_geocoded"].sum()),
            "geocoded_pct": round(100.0 * units["flag_geocoded"].mean(), 3)
            if len(units)
            else None,
            "high_confidence": int(units["flag_geocode_high_confidence"].sum()),
            "medium_confidence": int(units["flag_geocode_medium_confidence"].sum()),
            "low_confidence": int(units["flag_geocode_low_confidence"].sum()),
            "methods": {str(k): int(v) for k, v in methods.items()},
        },
        "transactions": {
            "rows": int(len(transactions)),
            "geocoded": int(transactions["latitude"].notna().sum()),
            "high_quality": int(transactions["geocode_quality"].eq("high").sum()),
            "medium_quality": int(transactions["geocode_quality"].eq("medium").sum()),
            "low_quality": int(transactions["geocode_quality"].eq("low").sum()),
            "multiple_sectors": int(
                transactions["flag_multiple_cnefe_sectors"].fillna(False).sum()
            ),
        },
        "methodology": {
            "reference": "CNEFE 2022 / IBGE",
            "sector_code": "COD_SETOR do endereço CNEFE, normalizado para 15 dígitos",
            "street_normalization": "ASCII, caixa alta, pontuação removida e tipo de logradouro removido",
            "high_confidence_threshold": 90,
            "note": (
                "Geocodificação fuzzy é sempre identificada por método e score. "
                "Mapas/modelos podem ser restritos a score >= 90."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Georreferencia o ITBI de Porto Alegre usando CNEFE 2022."
    )
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--transactions", type=Path, default=DEFAULT_TRANSACTIONS)
    parser.add_argument("--cnefe-dir", type=Path, default=DEFAULT_CNEFE_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Reconstrói o índice de endereços do CNEFE.",
    )
    parser.add_argument(
        "--allow-street-centroid",
        action="store_true",
        help=(
            "Para endereços sem número/match, permite usar centroide dos pontos "
            "CNEFE do mesmo logradouro+CEP (baixa confiança)."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    units_path = resolve_path(args.units)
    transactions_path = resolve_path(args.transactions)
    cnefe_dir = resolve_path(args.cnefe_dir)
    index_path = resolve_path(args.index)
    output_dir = resolve_path(args.output)

    if not units_path.exists() or not transactions_path.exists():
        print(
            "[erro] Bases limpas do ITBI não encontradas. Execute antes: "
            "python scripts/02_prepare_itbi.py",
            file=sys.stderr,
        )
        return 1

    try:
        units = pd.read_parquet(units_path)
        transactions = pd.read_parquet(transactions_path)

        cnefe_index = load_or_build_index(
            cnefe_dir=cnefe_dir,
            index_path=index_path,
            rebuild=args.rebuild_index,
        )

        units_geo = geocode_units(
            units,
            cnefe_index,
            allow_street_centroid=args.allow_street_centroid,
        )

        tx_geo = aggregate_transaction_geocodes(units_geo)
        transactions_geo = transactions.merge(
            tx_geo,
            on="transaction_id",
            how="left",
            validate="one_to_one",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        units_out = output_dir / "itbi_units_geocoded.parquet"
        tx_out = output_dir / "itbi_transactions_geocoded.parquet"
        summary_out = output_dir / "geocoding_summary.csv"
        quality_out = output_dir / "geocoding_quality.json"

        units_geo.to_parquet(units_out, index=False)
        transactions_geo.to_parquet(tx_out, index=False)
        build_summary(units_geo).to_csv(summary_out, index=False, encoding="utf-8")
        quality_out.write_text(
            json.dumps(
                quality_payload(units_geo, transactions_geo),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        print("\n[ok] Georreferenciamento concluído")
        print(
            f"     unidades: {int(units_geo['flag_geocoded'].sum()):,}/"
            f"{len(units_geo):,} "
            f"({100 * units_geo['flag_geocoded'].mean():.1f}%)"
        )
        print(
            f"     transações geocodificadas: "
            f"{int(transactions_geo['latitude'].notna().sum()):,}/"
            f"{len(transactions_geo):,}"
        )
        for path in (units_out, tx_out, summary_out, quality_out):
            print(f"     {path.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
