#!/usr/bin/env python3
"""Constrói o perfil socioeconômico de Porto Alegre com o Censo 2022/IBGE.

Entradas esperadas (geradas por scripts/downloads.py --groups censo):
    data/raw/ibge/censo2022/agregados_setor/*basico*.zip
    data/raw/ibge/censo2022/agregados_setor/*demografia*.zip
    data/raw/ibge/censo2022/renda_responsavel/*setores*renda*.zip

Opcionalmente, para perfil oficial por bairro:
    data/raw/ibge/censo2022/agregados_bairro/*basico*.zip
    data/raw/ibge/censo2022/agregados_bairro/*demografia*.zip
    data/raw/ibge/censo2022/renda_responsavel_bairro/*bairros*renda*.zip

Saídas:
    data/processed/census_sector_profile.parquet
    data/processed/census_neighborhood_profile.parquet
    data/processed/census_neighborhood_profile.csv
    data/processed/census_profile_quality.json
    data/processed/census_profile_metadata.json

Indicadores principais:
- população e domicílios;
- média de moradores por domicílio (proxy de tamanho domiciliar, não "família");
- sexo;
- estrutura etária e idade média aproximada;
- população 0-14, 15-29, 30-59 e 60+;
- densidade populacional, quando AREA_KM2 está disponível;
- rendimento nominal médio mensal da pessoa responsável com rendimento
  (V06004), explicitamente separado de renda domiciliar.

Valores protegidos pelo IBGE, como "x", são tratados como ausentes, nunca como zero.

Uso:
    python scripts/05_prepare_census.py
    python scripts/04_prepare_census.py --municipality 4314902

Dependências:
    pandas
    pyarrow
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CENSUS = PROJECT_ROOT / "data" / "raw" / "ibge" / "censo2022"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"
PORTO_ALEGRE_IBGE = "4314902"

BASIC_VARS = [f"V000{i}" for i in range(1, 8)]
DEMO_VARS = [f"V010{i:02d}" for i in range(6, 42)]
INCOME_VARS = [f"V0600{i}" for i in range(1, 6)]

AGE_GROUPS = {
    "pop_0_4": "V01031",
    "pop_5_9": "V01032",
    "pop_10_14": "V01033",
    "pop_15_19": "V01034",
    "pop_20_24": "V01035",
    "pop_25_29": "V01036",
    "pop_30_39": "V01037",
    "pop_40_49": "V01038",
    "pop_50_59": "V01039",
    "pop_60_69": "V01040",
    "pop_70_mais": "V01041",
}

AGE_MIDPOINTS = {
    "pop_0_4": 2.0,
    "pop_5_9": 7.0,
    "pop_10_14": 12.0,
    "pop_15_19": 17.0,
    "pop_20_24": 22.0,
    "pop_25_29": 27.0,
    "pop_30_39": 34.5,
    "pop_40_49": 44.5,
    "pop_50_59": 54.5,
    "pop_60_69": 64.5,
    # Faixa aberta. 75 é apenas ponto representativo para produzir uma
    # média aproximada; o campo final é nomeado explicitamente como "aprox".
    "pop_70_mais": 75.0,
}

SUPPRESSED_TOKENS = {
    "x",
    "X",
    "XX",
    "...",
    "-",
    "",
    "NA",
    "N/A",
    "null",
    "NULL",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_header(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_code(value: object, length: int | None = None) -> str | None:
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    if length is not None and len(digits) >= length:
        return digits[:length]
    return digits


def detect_delimiter(sample: bytes) -> str:
    text = sample.decode("utf-8-sig", errors="replace")
    try:
        return csv.Sniffer().sniff(text, delimiters=";,\t|").delimiter
    except csv.Error:
        return ";"


def find_zip(directory: Path, patterns: Iterable[str], required: bool = True) -> Path | None:
    if not directory.exists():
        if required:
            raise FileNotFoundError(f"Diretório não encontrado: {directory}")
        return None

    candidates = sorted(directory.glob("*.zip"))
    for pattern in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        matches = [p for p in candidates if regex.search(p.name)]
        if matches:
            return sorted(matches, key=lambda p: p.name)[-1]

    if required:
        raise FileNotFoundError(
            f"Nenhum ZIP compatível em {directory}. Padrões: {list(patterns)}"
        )
    return None


def inspect_zip_csv(zip_path: Path) -> tuple[str, str, dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.endswith("/")
        ]
        if not members:
            raise ValueError(f"{zip_path.name}: nenhum CSV encontrado.")

        member = sorted(members, key=lambda x: (len(x), x))[0]
        with archive.open(member) as stream:
            sample = stream.read(65536)
        delimiter = detect_delimiter(sample)

        with archive.open(member) as stream:
            header = pd.read_csv(
                stream,
                sep=delimiter,
                dtype="string",
                nrows=0,
                encoding="utf-8-sig",
                engine="python",
            )

    mapping = {col: normalize_header(col) for col in header.columns}
    return member, delimiter, mapping


def read_filtered_zip(
    zip_path: Path,
    key_field: str,
    value_vars: Iterable[str],
    municipality: str,
) -> pd.DataFrame:
    """Lê apenas Porto Alegre de um CSV nacional dentro de ZIP."""
    member, delimiter, mapping = inspect_zip_csv(zip_path)
    available = set(mapping.values())

    geography_candidates = {
        "CD_SETOR",
        "CD_BAIRRO",
        "NM_BAIRRO",
        "CD_MUN",
        "NM_MUN",
        "CD_UF",
        "NM_UF",
        "SITUACAO",
        "CD_SITUACAO",
        "AREA_KM2",
        "CD_DIST",
        "NM_DIST",
        "CD_SUBDIST",
        "NM_SUBDIST",
    }

    wanted = set(value_vars) | geography_candidates | {key_field}
    use_original = [
        original for original, normalized in mapping.items() if normalized in wanted
    ]

    if key_field not in available:
        raise ValueError(
            f"{zip_path.name}: campo-chave {key_field} ausente. "
            f"Campos: {sorted(available)}"
        )

    frames: list[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as stream:
            reader = pd.read_csv(
                stream,
                sep=delimiter,
                usecols=use_original,
                dtype="string",
                encoding="utf-8-sig",
                engine="python",
                chunksize=150_000,
                keep_default_na=True,
                na_values=list(SUPPRESSED_TOKENS),
            )

            for chunk in reader:
                chunk = chunk.rename(
                    columns={col: normalize_header(col) for col in chunk.columns}
                )

                if "CD_MUN" in chunk.columns:
                    mun = chunk["CD_MUN"].map(lambda x: normalize_code(x, 7))
                    chunk = chunk.loc[mun.eq(municipality)].copy()
                elif key_field == "CD_SETOR":
                    key = chunk[key_field].map(lambda x: normalize_code(x, 15))
                    chunk = chunk.loc[key.str.startswith(municipality, na=False)].copy()
                else:
                    key = chunk[key_field].map(normalize_code)
                    chunk = chunk.loc[key.str.startswith(municipality, na=False)].copy()

                if len(chunk):
                    frames.append(chunk)

    if not frames:
        raise ValueError(
            f"{zip_path.name}: nenhum registro encontrado para município {municipality}."
        )

    frame = pd.concat(frames, ignore_index=True, sort=False)

    if key_field == "CD_SETOR":
        frame[key_field] = frame[key_field].map(lambda x: normalize_code(x, 15))
    else:
        frame[key_field] = frame[key_field].map(normalize_code)

    if "CD_MUN" in frame.columns:
        frame["CD_MUN"] = frame["CD_MUN"].map(lambda x: normalize_code(x, 7))

    if "CD_BAIRRO" in frame.columns:
        frame["CD_BAIRRO"] = frame["CD_BAIRRO"].map(normalize_code)

    for column in value_vars:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "AREA_KM2" in frame.columns:
        frame["AREA_KM2"] = pd.to_numeric(frame["AREA_KM2"], errors="coerce")

    if frame[key_field].duplicated().any():
        duplicates = int(frame[key_field].duplicated(keep=False).sum())
        raise ValueError(
            f"{zip_path.name}: {duplicates} linhas com {key_field} duplicado "
            "após filtro municipal."
        )

    return frame


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    multiplier: float = 1.0,
) -> pd.Series:
    result = numerator / denominator
    result = result.where(denominator.gt(0))
    return result * multiplier


def complete_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    data = frame[columns].apply(pd.to_numeric, errors="coerce")
    return data.sum(axis=1, min_count=len(columns))


def approximate_mean_age(frame: pd.DataFrame) -> pd.Series:
    fields = list(AGE_GROUPS.keys())
    values = frame[fields].apply(pd.to_numeric, errors="coerce")
    complete = values.notna().all(axis=1)

    weighted = pd.Series(0.0, index=frame.index)
    total = pd.Series(0.0, index=frame.index)

    for field, midpoint in AGE_MIDPOINTS.items():
        weighted = weighted + values[field].fillna(0) * midpoint
        total = total + values[field].fillna(0)

    mean = weighted / total
    return mean.where(complete & total.gt(0))


def select_geo_columns(frame: pd.DataFrame, key_field: str) -> list[str]:
    preferred = [
        key_field,
        "CD_MUN",
        "NM_MUN",
        "CD_UF",
        "NM_UF",
        "CD_BAIRRO",
        "NM_BAIRRO",
        "SITUACAO",
        "CD_SITUACAO",
        "AREA_KM2",
        "CD_DIST",
        "NM_DIST",
        "CD_SUBDIST",
        "NM_SUBDIST",
    ]
    result: list[str] = []
    for column in preferred:
        if column in frame.columns and column not in result:
            result.append(column)
    return result


def build_profile(
    basic: pd.DataFrame,
    demo: pd.DataFrame,
    income: pd.DataFrame,
    key_field: str,
) -> pd.DataFrame:
    geo_cols = select_geo_columns(basic, key_field)
    base = basic[geo_cols + [c for c in BASIC_VARS if c in basic.columns]].copy()

    demo_keep = [key_field] + [c for c in DEMO_VARS if c in demo.columns]
    income_keep = [key_field] + [c for c in INCOME_VARS if c in income.columns]

    frame = (
        base.merge(demo[demo_keep], on=key_field, how="left", validate="one_to_one")
        .merge(
            income[income_keep],
            on=key_field,
            how="left",
            validate="one_to_one",
        )
    )

    frame["populacao_total"] = frame["V01006"].combine_first(frame["V0001"])
    frame["domicilios_total"] = frame["V0002"]
    frame["domicilios_particulares"] = frame["V0003"]
    frame["domicilios_coletivos"] = frame["V0004"]
    frame["media_moradores_domicilio"] = frame["V0005"]
    frame["pct_domicilios_ocupados_imputados"] = frame["V0006"]
    frame["domicilios_particulares_ocupados"] = frame["V0007"]

    frame["homens"] = frame["V01007"]
    frame["mulheres"] = frame["V01008"]
    frame["pct_homens"] = safe_ratio(frame["homens"], frame["populacao_total"], 100)
    frame["pct_mulheres"] = safe_ratio(frame["mulheres"], frame["populacao_total"], 100)

    for target, source in AGE_GROUPS.items():
        frame[target] = frame[source]

    frame["pop_0_14"] = complete_sum(
        frame, ["pop_0_4", "pop_5_9", "pop_10_14"]
    )
    frame["pop_15_29"] = complete_sum(
        frame, ["pop_15_19", "pop_20_24", "pop_25_29"]
    )
    frame["pop_30_59"] = complete_sum(
        frame, ["pop_30_39", "pop_40_49", "pop_50_59"]
    )
    frame["pop_60_mais"] = complete_sum(frame, ["pop_60_69", "pop_70_mais"])
    frame["pop_15_59"] = complete_sum(
        frame,
        [
            "pop_15_19",
            "pop_20_24",
            "pop_25_29",
            "pop_30_39",
            "pop_40_49",
            "pop_50_59",
        ],
    )

    for field in ("pop_0_14", "pop_15_29", "pop_30_59", "pop_60_mais"):
        frame[f"pct_{field}"] = safe_ratio(
            frame[field], frame["populacao_total"], 100
        )

    frame["idade_media_aprox"] = approximate_mean_age(frame)
    frame["razao_dependencia_0_14_60mais"] = safe_ratio(
        frame["pop_0_14"] + frame["pop_60_mais"],
        frame["pop_15_59"],
        100,
    )

    frame["responsaveis_dom_permanentes_ocupados"] = frame["V06001"]
    frame["moradores_dom_permanentes_ocupados"] = frame["V06002"]
    frame["variancia_moradores_dom_permanentes"] = frame["V06003"]
    frame["renda_media_responsavel_com_rendimento_rs"] = frame["V06004"]
    frame["variancia_renda_responsavel"] = frame["V06005"]

    frame["dp_moradores_dom_permanentes"] = frame[
        "variancia_moradores_dom_permanentes"
    ].map(lambda x: math.sqrt(x) if pd.notna(x) and x >= 0 else math.nan)
    frame["dp_renda_responsavel_rs"] = frame["variancia_renda_responsavel"].map(
        lambda x: math.sqrt(x) if pd.notna(x) and x >= 0 else math.nan
    )
    frame["cv_renda_responsavel"] = safe_ratio(
        frame["dp_renda_responsavel_rs"],
        frame["renda_media_responsavel_com_rendimento_rs"],
    )

    if "AREA_KM2" in frame.columns:
        frame["densidade_pop_km2"] = safe_ratio(
            frame["populacao_total"], frame["AREA_KM2"]
        )
    else:
        frame["densidade_pop_km2"] = pd.NA

    # QC interno: o básico e a demografia deveriam apontar totais muito próximos.
    frame["dif_pop_basico_demografia"] = frame["V0001"] - frame["V01006"]
    frame["flag_pop_divergente"] = (
        frame["dif_pop_basico_demografia"].abs().gt(0)
        & frame["V0001"].notna()
        & frame["V01006"].notna()
    )

    metric_order = [
        "populacao_total",
        "domicilios_total",
        "domicilios_particulares",
        "domicilios_coletivos",
        "domicilios_particulares_ocupados",
        "media_moradores_domicilio",
        "pct_domicilios_ocupados_imputados",
        "densidade_pop_km2",
        "homens",
        "mulheres",
        "pct_homens",
        "pct_mulheres",
        *AGE_GROUPS.keys(),
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
        "responsaveis_dom_permanentes_ocupados",
        "moradores_dom_permanentes_ocupados",
        "renda_media_responsavel_com_rendimento_rs",
        "dp_renda_responsavel_rs",
        "cv_renda_responsavel",
        "dif_pop_basico_demografia",
        "flag_pop_divergente",
    ]

    raw_order = BASIC_VARS + DEMO_VARS + INCOME_VARS
    ordered = []
    for column in geo_cols + metric_order + raw_order:
        if column in frame.columns and column not in ordered:
            ordered.append(column)
    ordered += [c for c in frame.columns if c not in ordered]

    return frame[ordered]


def load_level(
    level: str,
    municipality: str,
    required: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if level == "sector":
        key = "CD_SETOR"
        basic_dir = RAW_CENSUS / "agregados_setor"
        income_dir = RAW_CENSUS / "renda_responsavel"

        basic_zip = find_zip(
            basic_dir,
            [r"setores_basico.*\.zip$", r"basico.*\.zip$"],
            required=required,
        )
        demo_zip = find_zip(
            basic_dir,
            [r"setores_demografia.*\.zip$", r"demografia.*\.zip$"],
            required=required,
        )
        income_zip = find_zip(
            income_dir,
            [r"setores_renda_responsavel.*csv\.zip$", r"setores.*renda.*\.zip$"],
            required=required,
        )
    elif level == "neighborhood":
        key = "CD_BAIRRO"
        basic_dir = RAW_CENSUS / "agregados_bairro"
        income_dir = RAW_CENSUS / "renda_responsavel_bairro"

        basic_zip = find_zip(
            basic_dir,
            [r"bairros_basico.*\.zip$", r"basico.*\.zip$"],
            required=required,
        )
        demo_zip = find_zip(
            basic_dir,
            [r"bairros_demografia.*\.zip$", r"demografia.*\.zip$"],
            required=required,
        )
        income_zip = find_zip(
            income_dir,
            [r"bairros_renda_responsavel.*csv\.zip$", r"bairros.*renda.*\.zip$"],
            required=required,
        )
    else:
        raise ValueError(level)

    if basic_zip is None or demo_zip is None or income_zip is None:
        return None

    print(f"[{level}] básico:     {basic_zip.relative_to(PROJECT_ROOT)}")
    print(f"[{level}] demografia: {demo_zip.relative_to(PROJECT_ROOT)}")
    print(f"[{level}] renda:       {income_zip.relative_to(PROJECT_ROOT)}")

    basic = read_filtered_zip(basic_zip, key, BASIC_VARS, municipality)
    demo = read_filtered_zip(demo_zip, key, DEMO_VARS, municipality)
    income = read_filtered_zip(income_zip, key, INCOME_VARS, municipality)
    return basic, demo, income


def missing_rate(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or not len(frame):
        return None
    return round(100 * frame[column].isna().mean(), 3)


def quality_summary(
    sector: pd.DataFrame,
    neighborhood: pd.DataFrame | None,
    municipality: str,
) -> dict:
    payload = {
        "generated_at": utc_now_iso(),
        "municipality_code": municipality,
        "sector": {
            "rows": int(len(sector)),
            "population_total_sum": float(sector["populacao_total"].sum(min_count=1)),
            "households_total_sum": float(sector["domicilios_total"].sum(min_count=1)),
            "population_divergent_rows": int(
                sector["flag_pop_divergente"].fillna(False).sum()
            ),
            "missing_pct": {
                "populacao_total": missing_rate(sector, "populacao_total"),
                "media_moradores_domicilio": missing_rate(
                    sector, "media_moradores_domicilio"
                ),
                "idade_media_aprox": missing_rate(sector, "idade_media_aprox"),
                "renda_media_responsavel_com_rendimento_rs": missing_rate(
                    sector, "renda_media_responsavel_com_rendimento_rs"
                ),
            },
        },
    }

    if neighborhood is not None:
        payload["neighborhood"] = {
            "rows": int(len(neighborhood)),
            "population_total_sum": float(
                neighborhood["populacao_total"].sum(min_count=1)
            ),
            "missing_pct": {
                "idade_media_aprox": missing_rate(
                    neighborhood, "idade_media_aprox"
                ),
                "renda_media_responsavel_com_rendimento_rs": missing_rate(
                    neighborhood, "renda_media_responsavel_com_rendimento_rs"
                ),
            },
        }

    return payload


def metadata_payload() -> dict:
    return {
        "generated_at": utc_now_iso(),
        "source": "IBGE Censo Demográfico 2022 - Agregados por Setores Censitários",
        "variables": {
            "populacao_total": {
                "source": "V01006, com fallback V0001",
                "definition": "Quantidade de moradores / total de pessoas.",
            },
            "media_moradores_domicilio": {
                "source": "V0005",
                "definition": (
                    "Média de moradores em domicílios particulares ocupados. "
                    "É proxy de tamanho domiciliar e não deve ser chamada de tamanho "
                    "da família."
                ),
            },
            "idade_media_aprox": {
                "source": "V01031-V01041",
                "definition": (
                    "Média aproximada a partir dos pontos médios das faixas etárias."
                ),
                "assumption": (
                    "A faixa aberta de 70 anos ou mais usa 75 anos como ponto "
                    "representativo; por isso o indicador é explicitamente aproximado."
                ),
            },
            "pop_60_mais": {
                "source": "V01040 + V01041",
                "definition": "Pessoas de 60 a 69 anos mais pessoas de 70 anos ou mais.",
            },
            "renda_media_responsavel_com_rendimento_rs": {
                "source": "V06004",
                "definition": (
                    "Valor do rendimento nominal médio mensal das pessoas responsáveis "
                    "COM rendimento por domicílios particulares permanentes ocupados."
                ),
                "warning": (
                    "Não é renda domiciliar, renda per capita nem renda média de todos "
                    "os moradores."
                ),
            },
            "razao_dependencia_0_14_60mais": {
                "source": "V01031-V01041",
                "definition": "(população 0-14 + população 60+) / população 15-59 x 100.",
                "warning": (
                    "Indicador adaptado às faixas publicadas; não equivale à definição "
                    "convencional que pode usar 65 anos como corte superior."
                ),
            },
        },
        "confidentiality": (
            "Células publicadas como 'x' ou equivalentes são tratadas como ausentes "
            "(NaN), nunca como zero. Indicadores que dependem de todas as faixas ficam "
            "ausentes quando alguma faixa necessária estiver protegida."
        ),
        "neighborhood": (
            "O perfil por bairro usa diretamente os agregados oficiais por bairro do "
            "IBGE; não recalcula a renda média a partir dos setores."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara perfil socioeconômico do Censo 2022 para Porto Alegre."
    )
    parser.add_argument(
        "--municipality",
        default=PORTO_ALEGRE_IBGE,
        help="Código IBGE de 7 dígitos. Porto Alegre = 4314902.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    municipality = normalize_code(args.municipality, 7)
    if municipality is None or len(municipality) != 7:
        print("[erro] --municipality deve ter 7 dígitos.", file=sys.stderr)
        return 1

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sector_data = load_level("sector", municipality, required=True)
        assert sector_data is not None
        sector = build_profile(*sector_data, key_field="CD_SETOR")

        neighborhood_data = load_level(
            "neighborhood", municipality, required=False
        )
        neighborhood = (
            build_profile(*neighborhood_data, key_field="CD_BAIRRO")
            if neighborhood_data is not None
            else None
        )

        sector_path = output_dir / "census_sector_profile.parquet"
        sector.to_parquet(sector_path, index=False)

        neighborhood_path = output_dir / "census_neighborhood_profile.parquet"
        neighborhood_csv_path = output_dir / "census_neighborhood_profile.csv"
        if neighborhood is not None:
            neighborhood.to_parquet(neighborhood_path, index=False)
            neighborhood.to_csv(
                neighborhood_csv_path, index=False, encoding="utf-8"
            )
        else:
            print(
                "[warn] Agregados por bairro não encontrados. "
                "Rode novamente: python scripts/downloads.py --groups censo",
                file=sys.stderr,
            )

        quality_path = output_dir / "census_profile_quality.json"
        metadata_path = output_dir / "census_profile_metadata.json"
        quality_path.write_text(
            json.dumps(
                quality_summary(sector, neighborhood, municipality),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(metadata_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n[ok] Perfil socioeconômico construído")
        print(f"     setores: {len(sector):,}")
        print(
            f"     população (soma setorial): "
            f"{sector['populacao_total'].sum(min_count=1):,.0f}"
        )
        print(f"     {sector_path.relative_to(PROJECT_ROOT)}")
        if neighborhood is not None:
            print(f"     bairros: {len(neighborhood):,}")
            print(f"     {neighborhood_path.relative_to(PROJECT_ROOT)}")
            print(f"     {neighborhood_csv_path.relative_to(PROJECT_ROOT)}")
        print(f"     {quality_path.relative_to(PROJECT_ROOT)}")
        print(f"     {metadata_path.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
