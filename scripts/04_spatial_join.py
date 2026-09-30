#!/usr/bin/env python3
"""Associa transações geocodificadas às unidades espaciais de Porto Alegre.

Entrada:
    data/processed/itbi_transactions_geocoded.parquet
    data/raw/ibge/censo2022/malhas/RS_setores_CD2022.zip
    data/raw/porto_alegre/geometria/bairros/
    data/raw/porto_alegre/geometria/regioes_planejamento/
    data/raw/porto_alegre/geometria/regioes_op/

Saídas:
    data/processed/itbi_transactions_spatial.parquet
    data/processed/spatial_join_summary.json

A malha do IBGE é usada para obter o setor censitário definitivo. As camadas
municipais de bairro, Região de Planejamento e Região do OP são opcionais:
se alguma não estiver disponível, o script continua e registra a ausência.

Uso:
    python scripts/04_spatial_join.py

Dependências:
    pandas
    geopandas
    pyogrio
    pyarrow
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import unicodedata
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
DEFAULT_SECTORS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ibge"
    / "censo2022"
    / "malhas"
    / "RS_setores_CD2022.zip"
)
DEFAULT_BAIRROS = (
    PROJECT_ROOT / "data" / "raw" / "porto_alegre" / "geometria" / "bairros"
)
DEFAULT_RP = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "porto_alegre"
    / "geometria"
    / "regioes_planejamento"
)
DEFAULT_ROP = (
    PROJECT_ROOT / "data" / "raw" / "porto_alegre" / "geometria" / "regioes_op"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_spatial.parquet"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "data" / "processed" / "spatial_join_summary.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_header(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_digits(series: pd.Series, length: int | None = None) -> pd.Series:
    result = (
        series.astype("string")
        .str.replace(r"\D", "", regex=True)
        .replace("", pd.NA)
    )
    if length is not None:
        result = result.str.slice(0, length)
    return result


def extract_shapefile(zip_path: Path, temp_root: Path) -> Path:
    target = temp_root / zip_path.stem
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)

    shp = sorted(target.rglob("*.shp"))
    if not shp:
        raise ValueError(f"{zip_path.name}: nenhum .shp encontrado.")
    return shp[0]


def read_sector_mesh(zip_path: Path, municipality: str) -> gpd.GeoDataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Malha de setores não encontrada: {zip_path}. "
            "Execute: python scripts/downloads.py --groups geo"
        )

    with tempfile.TemporaryDirectory(prefix="sector_mesh_") as temp_dir:
        shp = extract_shapefile(zip_path, Path(temp_dir))
        frame = gpd.read_file(shp)

    # Não renomear a coluna de geometria: o GeoDataFrame mantém internamente
    # o nome da geometria ativa. Renomeá-la como uma coluna comum faz o objeto
    # perder a referência à geometria e quebra acessos como frame.crs/to_crs().
    geometry_col = frame.geometry.name
    rename_map = {
        c: normalize_header(c)
        for c in frame.columns
        if c != geometry_col
    }
    frame = frame.rename(columns=rename_map)

    if "CD_SETOR" not in frame.columns:
        raise ValueError(
            f"Malha IBGE sem CD_SETOR. Campos: {list(frame.columns)}"
        )

    frame["CD_SETOR"] = normalize_digits(frame["CD_SETOR"], 15)

    if "CD_MUN" in frame.columns:
        frame["CD_MUN"] = normalize_digits(frame["CD_MUN"], 7)
        frame = frame.loc[frame["CD_MUN"].eq(municipality)].copy()
    else:
        frame = frame.loc[
            frame["CD_SETOR"].str.startswith(municipality, na=False)
        ].copy()

    if frame.empty:
        raise ValueError(
            f"Nenhum setor encontrado para o município {municipality}."
        )
    if frame.crs is None:
        raise ValueError("Malha de setores sem CRS definido.")

    return frame[["CD_SETOR", "geometry"]].rename(
        columns={"CD_SETOR": "sector_final_code"}
    )


def find_vector_zip(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    zips = sorted(directory.glob("*.zip"))
    return zips[-1] if zips else None


def choose_label_column(columns: list[str], layer: str) -> str | None:
    priorities = {
        "bairro": [
            "NM_BAIRRO",
            "NOME_BAIRRO",
            "BAIRRO",
            "NOME",
            "NM",
            "NAME",
        ],
        "regiao_planejamento": [
            "NOME_REGIAO",
            "REGIAO",
            "NOME",
            "RP",
            "NM",
        ],
        "regiao_op": [
            "NOME_REGIAO",
            "REGIAO",
            "NOME",
            "ROP",
            "NM",
        ],
    }
    normalized = [normalize_header(c) for c in columns if c != "geometry"]

    for candidate in priorities[layer]:
        if candidate in normalized:
            return candidate

    keywords = {
        "bairro": ("BAIR", "NOME", "NM"),
        "regiao_planejamento": ("REG", "PLANEJ", "NOME", "RP"),
        "regiao_op": ("REG", "ORC", "NOME", "ROP"),
    }
    for keyword in keywords[layer]:
        for column in normalized:
            if keyword in column:
                return column
    return None


def choose_code_column(columns: list[str], label_col: str | None) -> str | None:
    normalized = [normalize_header(c) for c in columns if c != "geometry"]
    for keyword in ("COD", "CD_", "ID_", "ID", "NUM"):
        for column in normalized:
            if column != label_col and (
                column.startswith(keyword) or keyword in column
            ):
                return column
    return None


def read_optional_layer(
    directory: Path,
    layer: str,
) -> tuple[gpd.GeoDataFrame | None, str | None, str | None]:
    zip_path = find_vector_zip(directory)
    if zip_path is None:
        return None, None, None

    with tempfile.TemporaryDirectory(prefix=f"{layer}_") as temp_dir:
        shp = extract_shapefile(zip_path, Path(temp_dir))
        frame = gpd.read_file(shp)

    geometry_col = frame.geometry.name
    rename_map = {
        c: normalize_header(c)
        for c in frame.columns
        if c != geometry_col
    }
    frame = frame.rename(columns=rename_map)
    if frame.crs is None:
        raise ValueError(f"Camada {layer} sem CRS definido.")

    label = choose_label_column(list(frame.columns), layer)
    code = choose_code_column(list(frame.columns), label)

    keep = ["geometry"]
    if label:
        keep.insert(0, label)
    if code and code not in keep:
        keep.insert(0, code)

    return frame[keep].copy(), label, code


def make_points(transactions: pd.DataFrame) -> gpd.GeoDataFrame:
    lat = pd.to_numeric(transactions["latitude"], errors="coerce")
    lon = pd.to_numeric(transactions["longitude"], errors="coerce")

    valid = lat.between(-90, 90) & lon.between(-180, 180)

    points = gpd.GeoDataFrame(
        transactions.loc[valid].copy(),
        geometry=gpd.points_from_xy(lon.loc[valid], lat.loc[valid]),
        crs="EPSG:4326",
    )
    points["_row_id"] = points.index
    return points


def attach_polygon_attribute(
    result: pd.DataFrame,
    points_wgs84: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    output_label: str,
    source_label: str | None,
    output_code: str | None = None,
    source_code: str | None = None,
) -> pd.DataFrame:
    if points_wgs84.empty:
        result[output_label] = pd.NA
        if output_code:
            result[output_code] = pd.NA
        return result

    points = points_wgs84.to_crs(polygons.crs)
    joined = gpd.sjoin(points[["_row_id", "geometry"]], polygons, how="left", predicate="within")
    joined = joined.sort_values("_row_id").drop_duplicates("_row_id", keep="first")

    if source_label and source_label in joined.columns:
        result.loc[joined["_row_id"], output_label] = joined[source_label].to_numpy()
    else:
        result[output_label] = pd.NA

    if output_code:
        if source_code and source_code in joined.columns:
            result.loc[joined["_row_id"], output_code] = joined[source_code].to_numpy()
        else:
            result[output_code] = pd.NA

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Associa transações ITBI às unidades espaciais de Porto Alegre."
    )
    parser.add_argument("--itbi", type=Path, default=DEFAULT_ITBI)
    parser.add_argument("--sectors", type=Path, default=DEFAULT_SECTORS)
    parser.add_argument("--bairros", type=Path, default=DEFAULT_BAIRROS)
    parser.add_argument("--rp", type=Path, default=DEFAULT_RP)
    parser.add_argument("--rop", type=Path, default=DEFAULT_ROP)
    parser.add_argument("--municipality", default=PORTO_ALEGRE_IBGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()

    itbi_path = resolve(args.itbi)
    sectors_path = resolve(args.sectors)
    bairros_dir = resolve(args.bairros)
    rp_dir = resolve(args.rp)
    rop_dir = resolve(args.rop)
    output_path = resolve(args.output)
    summary_path = resolve(args.summary)

    if not itbi_path.exists():
        print(
            "[erro] ITBI georreferenciado ausente. Execute antes: "
            "python scripts/03_geocode_itbi.py",
            file=sys.stderr,
        )
        return 1

    try:
        transactions = pd.read_parquet(itbi_path)
        if "latitude" not in transactions or "longitude" not in transactions:
            raise ValueError("Base do ITBI não contém latitude/longitude.")

        result = transactions.copy()
        points = make_points(result)

        sectors = read_sector_mesh(sectors_path, str(args.municipality))
        result["sector_final_code"] = pd.NA
        result = attach_polygon_attribute(
            result,
            points,
            sectors,
            output_label="sector_final_code",
            source_label="sector_final_code",
        )
        result["sector_assignment_method"] = result["sector_final_code"].notna().map(
            {True: "spatial_final_mesh", False: "unmatched"}
        )

        layer_status: dict[str, str] = {}

        for directory, layer, out_label, out_code in [
            (bairros_dir, "bairro", "bairro_spatial", "bairro_spatial_code"),
            (
                rp_dir,
                "regiao_planejamento",
                "regiao_planejamento",
                "regiao_planejamento_code",
            ),
            (rop_dir, "regiao_op", "regiao_op", "regiao_op_code"),
        ]:
            try:
                polygons, label_col, code_col = read_optional_layer(directory, layer)
                if polygons is None:
                    result[out_label] = pd.NA
                    result[out_code] = pd.NA
                    layer_status[layer] = "not_found"
                    continue

                result = attach_polygon_attribute(
                    result,
                    points,
                    polygons,
                    output_label=out_label,
                    source_label=label_col,
                    output_code=out_code,
                    source_code=code_col,
                )
                layer_status[layer] = "ok"
            except Exception as exc:
                result[out_label] = pd.NA
                result[out_code] = pd.NA
                layer_status[layer] = f"error: {type(exc).__name__}: {exc}"

        result["flag_spatial_sector_matched"] = result["sector_final_code"].notna()
        result["flag_spatial_bairro_matched"] = result["bairro_spatial"].notna()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)

        total = len(result)
        geocoded = int(result["latitude"].notna().sum())
        sector_matched = int(result["flag_spatial_sector_matched"].sum())
        bairro_matched = int(result["flag_spatial_bairro_matched"].sum())

        summary = {
            "generated_at": utc_now_iso(),
            "transactions": total,
            "geocoded": geocoded,
            "sector_matched": sector_matched,
            "sector_matched_pct": round(100 * sector_matched / total, 3)
            if total
            else None,
            "bairro_matched": bairro_matched,
            "bairro_matched_pct": round(100 * bairro_matched / total, 3)
            if total
            else None,
            "optional_layers": layer_status,
            "output": str(output_path.relative_to(PROJECT_ROOT)),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n[ok] Associação espacial concluída")
        print(f"     transações: {total:,}")
        print(f"     geocodificadas: {geocoded:,}")
        print(f"     com setor definitivo: {sector_matched:,}")
        print(f"     com bairro espacial: {bairro_matched:,}")
        print(f"     {output_path.relative_to(PROJECT_ROOT)}")
        print(f"     {summary_path.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
