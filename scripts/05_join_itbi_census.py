#!/usr/bin/env python3
"""Integra ITBI georreferenciado ao Censo 2022 usando a malha DEFINITIVA de setores.

Entradas:
    data/processed/itbi_transactions_geocoded.parquet
    data/processed/census_sector_profile.parquet
    raw/ibge/censo2022/malhas/RS_setores_CD2022.zip

Saídas:
    data/processed/itbi_transactions_enriched.parquet
    data/processed/itbi_census_join_summary.json

Por que usar spatial join:
O CNEFE 2022 pode carregar o geocódigo do setor usado na operação de coleta,
inclusive códigos preliminares. Já os agregados censitários atuais são definitivos.
Por isso, a associação principal é feita pela coordenada do imóvel contra a malha
definitiva de setores. O código CNEFE é preservado para auditoria e só é usado
como fallback quando também existe explicitamente no perfil censitário definitivo.

Nenhuma linha é removida.

Uso:
    python scripts/05_join_itbi_census.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTO_ALEGRE_IBGE = "4314902"

DEFAULT_ITBI = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_geocoded.parquet"
)
DEFAULT_CENSUS = (
    PROJECT_ROOT / "data" / "processed" / "census_sector_profile.parquet"
)
DEFAULT_MESH = (
    PROJECT_ROOT
    / "raw"
    / "ibge"
    / "censo2022"
    / "malhas"
    / "RS_setores_CD2022.zip"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_enriched.parquet"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "data" / "processed" / "itbi_census_join_summary.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_sector(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 15)
        .replace("", pd.NA)
    )


def normalize_municipality(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 7)
        .replace("", pd.NA)
    )


def read_final_sector_mesh(zip_path: Path, municipality: str) -> gpd.GeoDataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Malha não encontrada: {zip_path}. "
            "Execute: python scripts/downloads.py --groups geo"
        )

    with tempfile.TemporaryDirectory(prefix="ibge_setores_") as temp_dir:
        temp = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp)

        shp_files = sorted(temp.rglob("*.shp"))
        if not shp_files:
            raise ValueError(f"{zip_path.name}: nenhum .shp encontrado.")

        # O pacote da UF contém a camada de setores; se houver mais de um SHP,
        # priorizamos o nome que contém "setor".
        shp_files.sort(
            key=lambda p: ("setor" not in p.name.lower(), len(p.name), p.name)
        )
        sectors = gpd.read_file(shp_files[0])

    sectors = sectors.rename(columns={c: c.upper() for c in sectors.columns})

    if "CD_SETOR" not in sectors.columns:
        raise ValueError(
            f"Malha sem CD_SETOR. Campos: {list(sectors.columns)}"
        )

    sectors["CD_SETOR"] = normalize_sector(sectors["CD_SETOR"])

    if "CD_MUN" in sectors.columns:
        sectors["CD_MUN"] = normalize_municipality(sectors["CD_MUN"])
        sectors = sectors.loc[sectors["CD_MUN"].eq(municipality)].copy()
    else:
        sectors = sectors.loc[
            sectors["CD_SETOR"].str.startswith(municipality, na=False)
        ].copy()

    if sectors.empty:
        raise ValueError(
            f"Nenhum setor da malha encontrado para município {municipality}."
        )

    if sectors.crs is None:
        raise ValueError("Malha de setores sem CRS definido.")

    return sectors[["CD_SETOR", "geometry"]].copy()


def assign_final_sector(
    transactions: pd.DataFrame,
    sectors: gpd.GeoDataFrame,
    census_keys: set[str],
) -> pd.DataFrame:
    result = transactions.copy()
    result["sector_final_code"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["sector_assignment_method"] = "unmatched"

    valid = (
        pd.to_numeric(result["latitude"], errors="coerce").between(-90, 90)
        & pd.to_numeric(result["longitude"], errors="coerce").between(-180, 180)
    )

    if valid.any():
        points = gpd.GeoDataFrame(
            result.loc[valid, []].copy(),
            geometry=gpd.points_from_xy(
                pd.to_numeric(result.loc[valid, "longitude"], errors="coerce"),
                pd.to_numeric(result.loc[valid, "latitude"], errors="coerce"),
            ),
            crs="EPSG:4326",
        )
        points["_row_id"] = points.index

        if points.crs != sectors.crs:
            points = points.to_crs(sectors.crs)

        joined = gpd.sjoin(
            points,
            sectors,
            how="left",
            predicate="within",
        )

        duplicate_rows = joined["_row_id"].duplicated(keep=False)
        if duplicate_rows.any():
            # Polígonos de setor não deveriam se sobrepor. Se houver ocorrência
            # residual, mantemos a primeira e registramos no resumo pelo número
            # de linhas duplicadas antes do corte.
            joined = joined.sort_values(["_row_id", "CD_SETOR"]).drop_duplicates(
                "_row_id", keep="first"
            )

        mapped = joined.set_index("_row_id")["CD_SETOR"]
        result.loc[mapped.index, "sector_final_code"] = mapped.astype("string")
        spatial_mask = result["sector_final_code"].notna()
        result.loc[spatial_mask, "sector_assignment_method"] = "spatial_final_mesh"

    # Fallback conservador: só usa COD_SETOR do CNEFE quando o código existe
    # exatamente na base definitiva do Censo.
    if "cnefe_sector_code" in result.columns:
        cnefe = normalize_sector(result["cnefe_sector_code"])
        fallback = (
            result["sector_final_code"].isna()
            & cnefe.notna()
            & cnefe.isin(census_keys)
        )
        result.loc[fallback, "sector_final_code"] = cnefe.loc[fallback]
        result.loc[fallback, "sector_assignment_method"] = "cnefe_code_fallback"

        result["cnefe_sector_code_normalized"] = cnefe
        result["flag_cnefe_sector_matches_final"] = (
            cnefe.notna()
            & result["sector_final_code"].notna()
            & cnefe.eq(result["sector_final_code"])
        )
    else:
        result["cnefe_sector_code_normalized"] = pd.NA
        result["flag_cnefe_sector_matches_final"] = False

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integra ITBI georreferenciado e Censo 2022 pela malha definitiva "
            "de setores censitários."
        )
    )
    parser.add_argument("--itbi", type=Path, default=DEFAULT_ITBI)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--municipality", default=PORTO_ALEGRE_IBGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    itbi_path = resolve(args.itbi)
    census_path = resolve(args.census)
    mesh_path = resolve(args.mesh)
    output_path = resolve(args.output)
    summary_path = resolve(args.summary)
    municipality = str(args.municipality)

    if not itbi_path.exists():
        print(
            "[erro] ITBI georreferenciado ausente. Execute "
            "python scripts/03_geocode_itbi.py",
            file=sys.stderr,
        )
        return 1

    if not census_path.exists():
        print(
            "[erro] Perfil censitário ausente. Execute "
            "python scripts/04_prepare_census.py",
            file=sys.stderr,
        )
        return 1

    try:
        itbi = pd.read_parquet(itbi_path)
        census = pd.read_parquet(census_path)

        if "latitude" not in itbi.columns or "longitude" not in itbi.columns:
            raise ValueError("ITBI sem latitude/longitude.")
        if "CD_SETOR" not in census.columns:
            raise ValueError("Censo sem coluna CD_SETOR.")

        census = census.copy()
        census["sector_join_key"] = normalize_sector(census["CD_SETOR"])
        if census["sector_join_key"].duplicated().any():
            raise ValueError("Perfil censitário contém CD_SETOR duplicado.")

        census_keys = set(census["sector_join_key"].dropna().astype(str))
        sectors = read_final_sector_mesh(mesh_path, municipality)

        assigned = assign_final_sector(itbi, sectors, census_keys)
        assigned["sector_join_key"] = normalize_sector(assigned["sector_final_code"])

        census_payload = census.rename(columns={"CD_SETOR": "census_sector_code"})

        enriched = assigned.merge(
            census_payload,
            on="sector_join_key",
            how="left",
            validate="many_to_one",
            suffixes=("", "_census"),
        )

        enriched["flag_census_matched"] = enriched["census_sector_code"].notna()
        enriched["flag_geocode_usable"] = enriched["geocode_quality"].isin(
            ["high", "medium"]
        )
        enriched["flag_ready_for_spatial_analysis"] = (
            enriched["flag_geocode_usable"] & enriched["flag_census_matched"]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(output_path, index=False)

        total = len(enriched)
        geocoded = int(enriched["latitude"].notna().sum())
        spatial = int(
            enriched["sector_assignment_method"].eq("spatial_final_mesh").sum()
        )
        fallback = int(
            enriched["sector_assignment_method"].eq("cnefe_code_fallback").sum()
        )
        census_matched = int(enriched["flag_census_matched"].sum())
        ready = int(enriched["flag_ready_for_spatial_analysis"].sum())

        summary = {
            "generated_at": utc_now_iso(),
            "transactions": total,
            "geocoded": geocoded,
            "geocoded_pct": round(100 * geocoded / total, 3) if total else None,
            "sector_assigned_spatial_final_mesh": spatial,
            "sector_assigned_cnefe_fallback": fallback,
            "sector_unmatched": total - spatial - fallback,
            "cnefe_sector_matches_final": int(
                enriched["flag_cnefe_sector_matches_final"].fillna(False).sum()
            ),
            "census_matched": census_matched,
            "census_matched_pct": (
                round(100 * census_matched / total, 3) if total else None
            ),
            "ready_for_spatial_analysis": ready,
            "ready_for_spatial_analysis_pct": (
                round(100 * ready / total, 3) if total else None
            ),
            "primary_join": (
                "latitude/longitude -> malha definitiva de setores -> CD_SETOR"
            ),
            "fallback_join": (
                "CNEFE COD_SETOR somente quando o código existe no Censo definitivo"
            ),
            "output": str(output_path.relative_to(PROJECT_ROOT)),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n[ok] ITBI + Censo integrados pela malha definitiva")
        print(f"     transações: {total:,}")
        print(f"     geocodificadas: {geocoded:,}")
        print(f"     setor por spatial join: {spatial:,}")
        print(f"     setor por fallback CNEFE: {fallback:,}")
        print(f"     com perfil censitário: {census_matched:,}")
        print(f"     prontas para análise espacial: {ready:,}")
        print(f"     {output_path.relative_to(PROJECT_ROOT)}")
        print(f"     {summary_path.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
