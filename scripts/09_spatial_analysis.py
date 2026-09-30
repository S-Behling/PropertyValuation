#!/usr/bin/env python3
"""Analisa padrões espaciais do mercado imobiliário de Porto Alegre.

Entradas:
    data/processed/itbi_transactions_qc.parquet
    data/processed/market_indicators_sector_year.parquet
    data/processed/census_sector_profile.parquet
    data/raw/ibge/censo2022/malhas/RS_setores_CD2022.zip

Saídas:
    data/processed/spatial_global_moran.csv
    data/processed/spatial_local_moran_value.parquet
    data/processed/spatial_concentration.csv
    data/processed/spatial_analysis_metadata.json
    outputs/maps/sector_value_m2_YYYY.png
    outputs/maps/sector_transactions_YYYY.png
    outputs/maps/transaction_heatmap_YYYY.png
    outputs/maps/local_moran_value_YYYY.png
    outputs/maps/census_income_sector_2022.png
    outputs/maps/census_age_sector_2022.png
    outputs/maps/census_household_size_sector_2022.png
    outputs/figures/global_moran_by_year.png

Metodologia:
- processa a malha definitiva de setores censitários do Censo 2022;
- analisa valor/m² somente em setores com amostra mínima de transações;
- processa pesos espaciais de contiguidade Queen;
- analisa autocorrelação espacial global com Moran's I;
- analisa clusters locais com Local Moran/LISA e pseudo-p de permutação;
- processa mapas anuais com escala comum para permitir comparação temporal;
- processa heatmaps de transações em coordenadas projetadas;
- analisa concentração de transações entre setores;
- processa mapas socioeconômicos do Censo 2022 como contexto estático.

Observação:
    base_de_calculo e valor_m2_base são medidas derivadas da base tributária
    publicada do ITBI; não são tratados como preço efetivo de venda.

Uso:
    python scripts/09_spatial_analysis.py
    python scripts/09_spatial_analysis.py --min-transactions 5 --permutations 999

Dependências:
    pandas
    numpy
    geopandas
    matplotlib
    scipy
    libpysal
    esda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen
from matplotlib.colors import Normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTO_ALEGRE_IBGE = "4314902"

DEFAULT_TRANSACTIONS = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_qc.parquet"
)
DEFAULT_SECTOR_INDICATORS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_indicators_sector_year.parquet"
)
DEFAULT_CENSUS = (
    PROJECT_ROOT / "data" / "processed" / "census_sector_profile.parquet"
)
DEFAULT_SECTOR_MESH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ibge"
    / "censo2022"
    / "malhas"
    / "RS_setores_CD2022.zip"
)

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MAPS_DIR = PROJECT_ROOT / "outputs" / "maps"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_METADATA = (
    PROJECT_ROOT / "data" / "processed" / "spatial_analysis_metadata.json"
)

DEFAULT_MIN_TRANSACTIONS = 5
DEFAULT_PERMUTATIONS = 999
DEFAULT_SIGNIFICANCE = 0.05
DEFAULT_SEED = 42
DEFAULT_HEXBIN_GRIDSIZE = 60

VALUE_METRIC = "valor_m2_mediano_real"
VOLUME_METRIC = "n_transacoes"

CENSUS_MAP_SPECS = {
    "renda_media_responsavel_com_rendimento_rs": {
        "filename": "census_income_sector_2022.png",
        "title": (
            "Rendimento nominal médio mensal da pessoa responsável "
            "com rendimento — Censo 2022"
        ),
        "legend": "R$ nominais de 2022",
    },
    "idade_media_aprox": {
        "filename": "census_age_sector_2022.png",
        "title": "Idade média aproximada — Censo 2022",
        "legend": "anos",
    },
    "media_moradores_domicilio": {
        "filename": "census_household_size_sector_2022.png",
        "title": "Média de moradores por domicílio — Censo 2022",
        "legend": "moradores",
    },
}


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


def extract_shapefile(zip_path: Path, temp_root: Path) -> Path:
    """Processa a extração temporária e seleciona a camada de setores."""
    target = temp_root / zip_path.stem
    target.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)

    shapefiles = sorted(target.rglob("*.shp"))
    if not shapefiles:
        raise ValueError(f"{zip_path.name}: nenhum arquivo .shp encontrado.")

    # Analisa os nomes disponíveis e prioriza a camada explicitamente setorial.
    shapefiles.sort(
        key=lambda path: (
            "setor" not in path.name.lower(),
            len(path.name),
            path.name,
        )
    )
    return shapefiles[0]


def read_sector_mesh(
    zip_path: Path,
    municipality: str,
) -> gpd.GeoDataFrame:
    """Processa a malha definitiva e mantém somente Porto Alegre."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Malha de setores não encontrada: {zip_path}. "
            "Execute: python scripts/downloads.py --groups geo"
        )

    with tempfile.TemporaryDirectory(prefix="spatial_analysis_") as temp_dir:
        shp_path = extract_shapefile(zip_path, Path(temp_dir))
        sectors = gpd.read_file(shp_path)

    geometry_column = sectors.geometry.name
    rename_map = {
        column: str(column).strip().upper()
        for column in sectors.columns
        if column != geometry_column
    }
    sectors = sectors.rename(columns=rename_map)

    if "CD_SETOR" not in sectors.columns:
        raise ValueError(
            f"Malha sem CD_SETOR. Campos encontrados: {list(sectors.columns)}"
        )
    if sectors.crs is None:
        raise ValueError("Malha de setores sem CRS definido.")

    sectors["sector_final_code"] = normalize_sector_code(sectors["CD_SETOR"])

    if "CD_MUN" in sectors.columns:
        municipality_code = normalize_sector_code(sectors["CD_MUN"]).str.slice(0, 7)
        sectors = sectors.loc[municipality_code.eq(municipality)].copy()
    else:
        sectors = sectors.loc[
            sectors["sector_final_code"].str.startswith(municipality, na=False)
        ].copy()

    if sectors.empty:
        raise ValueError(
            f"Nenhum setor encontrado para o município {municipality}."
        )

    if sectors["sector_final_code"].duplicated().any():
        raise ValueError("Malha contém códigos de setor duplicados.")

    return sectors[["sector_final_code", geometry_column]].rename_geometry(
        geometry_column
    )


def load_sector_indicators(path: Path) -> pd.DataFrame:
    """Processa indicadores setor-ano e analisa as colunas obrigatórias."""
    if not path.exists():
        raise FileNotFoundError(
            f"Indicadores setoriais não encontrados: {path}. "
            "Execute antes: python scripts/08_market_indicators.py"
        )

    frame = pd.read_parquet(path)
    required = {
        "ano_analise",
        "sector_final_code",
        "n_transacoes",
        VALUE_METRIC,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Indicadores setoriais sem colunas obrigatórias: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["sector_final_code"] = normalize_sector_code(
        frame["sector_final_code"]
    )
    frame["ano_analise"] = pd.to_numeric(
        frame["ano_analise"],
        errors="coerce",
    ).astype("Int64")

    duplicated = frame.duplicated(
        ["ano_analise", "sector_final_code"],
        keep=False,
    )
    if duplicated.any():
        count = int(duplicated.sum())
        raise ValueError(
            f"Indicadores setoriais contêm {count} linhas ano-setor duplicadas."
        )

    return frame


def load_transactions(path: Path) -> pd.DataFrame:
    """Processa transações auditadas para os mapas de densidade."""
    if not path.exists():
        raise FileNotFoundError(
            f"Base auditada não encontrada: {path}. "
            "Execute antes: python scripts/07_quality_control.py"
        )

    frame = pd.read_parquet(path)
    required = {
        "source_year",
        "latitude",
        "longitude",
        "flag_use_spatial_analysis",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Base auditada sem colunas obrigatórias: {sorted(missing)}"
        )

    frame = frame.copy()

    # Processa o ano analítico a partir da data corrigida quando disponível.
    if "data_correcao_monetaria" in frame.columns:
        date = pd.to_datetime(
            frame["data_correcao_monetaria"],
            errors="coerce",
        )
        frame["ano_analise"] = date.dt.year.astype("Int64")
    else:
        frame["ano_analise"] = pd.Series(
            pd.NA,
            index=frame.index,
            dtype="Int64",
        )

    source_year = pd.to_numeric(
        frame["source_year"],
        errors="coerce",
    ).astype("Int64")
    frame["ano_analise"] = frame["ano_analise"].combine_first(source_year)

    return frame


def load_census(path: Path) -> pd.DataFrame:
    """Processa variáveis socioeconômicas do Censo para mapas contextuais."""
    if not path.exists():
        raise FileNotFoundError(
            f"Perfil censitário não encontrado: {path}. "
            "Execute antes: python scripts/05_prepare_census.py"
        )

    frame = pd.read_parquet(path)
    if "CD_SETOR" not in frame.columns:
        raise ValueError("Perfil censitário sem CD_SETOR.")

    frame = frame.copy()
    frame["sector_final_code"] = normalize_sector_code(frame["CD_SETOR"])

    selected = ["sector_final_code"]
    selected.extend(
        column
        for column in CENSUS_MAP_SPECS
        if column in frame.columns
    )

    return frame[selected].copy()


def build_year_geodata(
    sectors: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
    year: int,
) -> gpd.GeoDataFrame:
    """Processa a malha completa com os indicadores de um único ano."""
    yearly = indicators.loc[indicators["ano_analise"].eq(year)].copy()

    payload_columns = [
        "sector_final_code",
        "n_transacoes",
        VALUE_METRIC,
    ]
    if "flag_ano_incompleto" in yearly.columns:
        payload_columns.append("flag_ano_incompleto")

    return sectors.merge(
        yearly[payload_columns],
        on="sector_final_code",
        how="left",
        validate="one_to_one",
    )


def common_value_scale(
    indicators: pd.DataFrame,
    min_transactions: int,
) -> tuple[float, float]:
    """Analisa limites robustos comuns para comparar mapas entre anos."""
    valid = indicators.loc[
        pd.to_numeric(indicators["n_transacoes"], errors="coerce").ge(
            min_transactions
        )
    ]
    values = pd.to_numeric(valid[VALUE_METRIC], errors="coerce").dropna()

    if values.empty:
        raise ValueError("Não há valores setoriais suficientes para mapear.")

    lower = float(values.quantile(0.02))
    upper = float(values.quantile(0.98))

    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        lower = float(values.min())
        upper = float(values.max())

    return lower, upper


def common_volume_scale(
    indicators: pd.DataFrame,
) -> tuple[float, float]:
    """Analisa limites comuns do volume de transações entre anos."""
    counts = pd.to_numeric(
        indicators["n_transacoes"],
        errors="coerce",
    ).dropna()

    if counts.empty:
        return 0.0, 1.0

    upper = float(counts.quantile(0.98))
    return 0.0, max(1.0, upper)


def plot_sector_metric(
    geodata: gpd.GeoDataFrame,
    column: str,
    title: str,
    legend_label: str,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    minimum_transactions: int | None = None,
) -> None:
    """Processa um mapa coroplético setorial com layout consistente."""
    frame = geodata.copy()

    if minimum_transactions is not None:
        counts = pd.to_numeric(frame["n_transacoes"], errors="coerce")
        frame.loc[counts.lt(minimum_transactions), column] = np.nan

    fig, ax = plt.subplots(figsize=(9, 9))

    frame.plot(
        column=column,
        ax=ax,
        cmap="viridis",
        linewidth=0.08,
        edgecolor="white",
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={"label": legend_label, "shrink": 0.72},
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "label": "sem amostra suficiente",
        },
    )

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_queen_weights(
    geodata: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, Queen]:
    """Processa pesos Queen e remove ilhas da inferência espacial."""
    frame = geodata.copy()
    frame = frame.set_index("sector_final_code", drop=False)

    weights = Queen.from_dataframe(frame, use_index=True)

    # Analisa ilhas porque setores sem vizinhos não contribuem ao Moran.
    if weights.islands:
        frame = frame.drop(index=list(weights.islands))
        weights = Queen.from_dataframe(frame, use_index=True)

    if len(frame) < 3:
        raise ValueError("Menos de três setores conectados para Moran.")

    weights.transform = "r"
    return frame, weights


def prepare_value_moran_frame(
    sectors: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
    year: int,
    min_transactions: int,
) -> gpd.GeoDataFrame:
    """Processa setores com amostra mínima para autocorrelação de valor/m²."""
    frame = build_year_geodata(sectors, indicators, year)

    counts = pd.to_numeric(frame["n_transacoes"], errors="coerce")
    values = pd.to_numeric(frame[VALUE_METRIC], errors="coerce")

    frame[VALUE_METRIC] = values
    frame = frame.loc[
        counts.ge(min_transactions) & values.notna()
    ].copy()

    return frame


def prepare_volume_moran_frame(
    sectors: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
    year: int,
) -> gpd.GeoDataFrame:
    """Processa todos os setores e representa volume como log(1 + transações)."""
    frame = build_year_geodata(sectors, indicators, year)
    counts = pd.to_numeric(frame["n_transacoes"], errors="coerce").fillna(0)

    frame["log_transacoes"] = np.log1p(counts)
    return frame


def calculate_global_moran(
    geodata: gpd.GeoDataFrame,
    metric: str,
    year: int,
    permutations: int,
    seed: int,
    min_transactions: int | None,
    incomplete_year: bool,
) -> dict[str, Any]:
    """Analisa autocorrelação global com pesos Queen padronizados por linha."""
    frame, weights = _build_queen_weights(geodata)

    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{metric}: valores não finitos após filtro.")

    if np.nanstd(values) == 0:
        raise ValueError(f"{metric}: variância zero; Moran não é definido.")

    # Processa a semente antes das permutações para resultados reproduzíveis.
    np.random.seed(seed)
    moran = Moran(
        values,
        weights,
        permutations=permutations,
    )

    return {
        "ano_analise": year,
        "metric": metric,
        "n_setores": int(len(frame)),
        "min_transacoes_setor": min_transactions,
        "moran_i": float(moran.I),
        "moran_expected": float(moran.EI),
        "moran_z_sim": float(moran.z_sim),
        "moran_p_sim": float(moran.p_sim),
        "permutations": permutations,
        "flag_ano_incompleto": bool(incomplete_year),
    }


def calculate_local_moran(
    geodata: gpd.GeoDataFrame,
    year: int,
    permutations: int,
    significance: float,
    seed: int,
) -> gpd.GeoDataFrame:
    """Analisa clusters locais LISA para o valor/m² setorial."""
    frame, weights = _build_queen_weights(geodata)
    values = pd.to_numeric(
        frame[VALUE_METRIC],
        errors="coerce",
    ).to_numpy(dtype=float)

    local = Moran_Local(
        values,
        weights,
        permutations=permutations,
        seed=seed,
    )

    result = frame.copy()
    result["local_moran_i"] = local.Is
    result["local_moran_p_sim"] = local.p_sim
    result["local_moran_q"] = local.q

    significant = result["local_moran_p_sim"].le(significance)
    cluster_map = {
        1: "HH",
        2: "LH",
        3: "LL",
        4: "HL",
    }
    result["lisa_cluster"] = "NS"
    result.loc[significant, "lisa_cluster"] = (
        result.loc[significant, "local_moran_q"]
        .map(cluster_map)
        .fillna("NS")
    )
    result["ano_analise"] = year
    result["significance_level"] = significance

    return result


def plot_local_moran(
    sectors: gpd.GeoDataFrame,
    local: gpd.GeoDataFrame,
    year: int,
    output_path: Path,
) -> None:
    """Processa mapa de clusters LISA para o ano selecionado."""
    payload = local[
        ["sector_final_code", "lisa_cluster"]
    ].reset_index(drop=True)

    frame = sectors.merge(
        payload,
        on="sector_final_code",
        how="left",
        validate="one_to_one",
    )
    frame["lisa_cluster"] = frame["lisa_cluster"].fillna("Sem amostra")

    colors = {
        "HH": "#b2182b",
        "LL": "#2166ac",
        "HL": "#ef8a62",
        "LH": "#67a9cf",
        "NS": "#d9d9d9",
        "Sem amostra": "#f2f2f2",
    }

    fig, ax = plt.subplots(figsize=(9, 9))

    for category in ("HH", "LL", "HL", "LH", "NS", "Sem amostra"):
        subset = frame.loc[frame["lisa_cluster"].eq(category)]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            color=colors[category],
            linewidth=0.08,
            edgecolor="white",
            label=category,
        )

    ax.set_title(
        f"Local Moran/LISA — valor real mediano por m² — {year}\n"
        "HH/LL = clusters; HL/LH = outliers espaciais"
    )
    ax.set_axis_off()
    ax.legend(
        title="Cluster",
        loc="lower left",
        frameon=True,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def transactions_geodata(
    transactions: pd.DataFrame,
    year: int,
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """Processa pontos válidos da amostra espacial em CRS projetado."""
    spatial_flag = (
        transactions["flag_use_spatial_analysis"]
        .astype("boolean")
        .fillna(False)
    )

    year_mask = transactions["ano_analise"].eq(year)
    latitude = pd.to_numeric(transactions["latitude"], errors="coerce")
    longitude = pd.to_numeric(transactions["longitude"], errors="coerce")

    valid_coordinates = (
        latitude.between(-90, 90)
        & longitude.between(-180, 180)
    )

    subset = transactions.loc[
        spatial_flag & year_mask & valid_coordinates
    ].copy()

    points = gpd.GeoDataFrame(
        subset,
        geometry=gpd.points_from_xy(
            pd.to_numeric(subset["longitude"], errors="coerce"),
            pd.to_numeric(subset["latitude"], errors="coerce"),
        ),
        crs="EPSG:4326",
    )

    return points.to_crs(target_crs)


def plot_transaction_heatmap(
    transactions: pd.DataFrame,
    sectors: gpd.GeoDataFrame,
    year: int,
    output_path: Path,
    gridsize: int,
) -> None:
    """Processa heatmap hexagonal da concentração de transações."""
    points = transactions_geodata(
        transactions,
        year,
        sectors.crs,
    )

    if points.empty:
        return

    boundary = sectors.dissolve()

    fig, ax = plt.subplots(figsize=(9, 9))
    boundary.boundary.plot(
        ax=ax,
        linewidth=0.8,
        color="black",
    )

    x = points.geometry.x.to_numpy()
    y = points.geometry.y.to_numpy()

    density = ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        mincnt=1,
        bins="log",
        cmap="inferno",
        linewidths=0,
    )
    colorbar = fig.colorbar(density, ax=ax, shrink=0.72)
    colorbar.set_label("densidade de transações (escala log)")

    ax.set_title(f"Concentração espacial das transações — {year}")
    ax.set_axis_off()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def gini(values: np.ndarray) -> float:
    """Analisa desigualdade da distribuição espacial das transações."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    array = array[array >= 0]

    if len(array) == 0 or array.sum() == 0:
        return np.nan

    array = np.sort(array)
    n = len(array)
    cumulative = np.cumsum(array)
    return float(
        (n + 1 - 2 * (cumulative.sum() / cumulative[-1])) / n
    )


def build_concentration_table(
    sectors: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
) -> pd.DataFrame:
    """Analisa concentração anual do volume de transações entre setores."""
    rows: list[dict[str, Any]] = []

    for year in sorted(indicators["ano_analise"].dropna().astype(int).unique()):
        geodata = build_year_geodata(sectors, indicators, int(year))
        counts = pd.to_numeric(
            geodata["n_transacoes"],
            errors="coerce",
        ).fillna(0)

        total = float(counts.sum())
        shares = counts / total if total > 0 else counts * np.nan
        hhi = float((shares**2).sum()) if total > 0 else np.nan

        sorted_counts = counts.sort_values(ascending=False)
        top10_share = (
            float(100 * sorted_counts.head(10).sum() / total)
            if total > 0
            else np.nan
        )

        top10pct_n = max(1, math.ceil(0.10 * len(sorted_counts)))
        top10pct_share = (
            float(
                100
                * sorted_counts.head(top10pct_n).sum()
                / total
            )
            if total > 0
            else np.nan
        )

        rows.append(
            {
                "ano_analise": int(year),
                "n_setores_malha": int(len(counts)),
                "n_setores_com_transacao": int(counts.gt(0).sum()),
                "n_transacoes_espaciais": int(total),
                "hhi_transacoes_setor": hhi,
                "setores_efetivos_hhi": (
                    float(1 / hhi)
                    if pd.notna(hhi) and hhi > 0
                    else np.nan
                ),
                "gini_transacoes_setor": gini(counts.to_numpy()),
                "participacao_top10_setores_pct": top10_share,
                "participacao_top10pct_setores_pct": top10pct_share,
            }
        )

    return pd.DataFrame(rows)


def merge_census_geodata(
    sectors: gpd.GeoDataFrame,
    census: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Processa a associação espacial entre malha e perfil censitário."""
    return sectors.merge(
        census,
        on="sector_final_code",
        how="left",
        validate="one_to_one",
    )


def plot_census_maps(
    sectors: gpd.GeoDataFrame,
    census: pd.DataFrame,
    maps_dir: Path,
) -> list[str]:
    """Processa mapas socioeconômicos estáticos do Censo 2022."""
    geodata = merge_census_geodata(sectors, census)
    generated: list[str] = []

    for column, spec in CENSUS_MAP_SPECS.items():
        if column not in geodata.columns:
            continue

        values = pd.to_numeric(geodata[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue

        lower = float(valid.quantile(0.02))
        upper = float(valid.quantile(0.98))

        output_path = maps_dir / spec["filename"]
        plot_sector_metric(
            geodata=geodata.assign(**{column: values}),
            column=column,
            title=spec["title"],
            legend_label=spec["legend"],
            output_path=output_path,
            vmin=lower,
            vmax=upper,
        )
        generated.append(str(output_path.relative_to(PROJECT_ROOT)))

    return generated


def plot_global_moran_series(
    moran_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Processa série temporal do Moran global para as métricas analisadas."""
    if moran_table.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for metric, group in moran_table.groupby("metric"):
        ordered = group.sort_values("ano_analise")
        ax.plot(
            ordered["ano_analise"],
            ordered["moran_i"],
            marker="o",
            label=metric,
        )

    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Moran's I")
    ax.set_title("Autocorrelação espacial global por ano")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def determine_incomplete_years(
    indicators: pd.DataFrame,
) -> set[int]:
    """Analisa quais anos estão explicitamente marcados como incompletos."""
    if "flag_ano_incompleto" not in indicators.columns:
        return {datetime.now().year}

    incomplete = indicators.loc[
        indicators["flag_ano_incompleto"].fillna(False),
        "ano_analise",
    ]
    return set(incomplete.dropna().astype(int).unique())


def latest_complete_year(
    years: list[int],
    incomplete_years: set[int],
) -> int:
    """Processa o ano completo mais recente para o mapa LISA principal."""
    complete = [year for year in years if year not in incomplete_years]
    if complete:
        return max(complete)

    return max(years)


def build_metadata(
    years: list[int],
    incomplete_years: set[int],
    min_transactions: int,
    permutations: int,
    significance: float,
    local_year: int,
    moran_table: pd.DataFrame,
    concentration: pd.DataFrame,
    generated_maps: list[str],
) -> dict[str, Any]:
    """Analisa e documenta as premissas da etapa espacial."""
    return {
        "generated_at": utc_now_iso(),
        "years": years,
        "incomplete_years": sorted(incomplete_years),
        "local_moran_reference_year": local_year,
        "methodology": {
            "weights": "Queen contiguity, row-standardized",
            "global_moran_permutations": permutations,
            "local_moran_permutations": permutations,
            "local_moran_significance": significance,
            "value_metric": VALUE_METRIC,
            "minimum_transactions_for_value_moran": min_transactions,
            "volume_metric": "log1p(n_transacoes), incluindo setores com zero",
            "heatmap": "hexbin em CRS projetado",
            "census_context_year": 2022,
        },
        "quality": {
            "global_moran_rows": int(len(moran_table)),
            "concentration_rows": int(len(concentration)),
            "generated_maps": len(generated_maps),
        },
        "notes": [
            (
                "Analisa valor/m² apenas em setores com amostra mínima para "
                "reduzir instabilidade de medianas setoriais."
            ),
            (
                "Processa o volume espacial com setores sem transação igual a "
                "zero para não eliminar áreas sem atividade observada."
            ),
            (
                "Analisa 2026 como período incompleto quando essa flag estiver "
                "presente nos indicadores da etapa 08."
            ),
            (
                "Processa variáveis do Censo 2022 como contexto espacial "
                "estático; não as interpreta como perfil anual de 2020-2026."
            ),
            (
                "Analisa base_de_calculo como base tributária do ITBI e não "
                "como preço efetivo de venda."
            ),
        ],
        "maps": generated_maps,
    }


def parse_args() -> argparse.Namespace:
    """Processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Analisa padrões espaciais do mercado imobiliário."
    )
    parser.add_argument(
        "--transactions",
        type=Path,
        default=DEFAULT_TRANSACTIONS,
    )
    parser.add_argument(
        "--sector-indicators",
        type=Path,
        default=DEFAULT_SECTOR_INDICATORS,
    )
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument(
        "--sector-mesh",
        type=Path,
        default=DEFAULT_SECTOR_MESH,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--municipality",
        default=PORTO_ALEGRE_IBGE,
    )
    parser.add_argument(
        "--min-transactions",
        type=int,
        default=DEFAULT_MIN_TRANSACTIONS,
        help="Amostra mínima do setor para mapas/Moran de valor/m².",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=DEFAULT_SIGNIFICANCE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--hexbin-gridsize",
        type=int,
        default=DEFAULT_HEXBIN_GRIDSIZE,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Analisa parâmetros estatísticos e cartográficos da execução."""
    if args.min_transactions < 1:
        raise ValueError("--min-transactions deve ser pelo menos 1.")
    if args.permutations < 99:
        raise ValueError("--permutations deve ser pelo menos 99.")
    if not 0 < args.significance < 1:
        raise ValueError("--significance deve estar entre 0 e 1.")
    if args.hexbin_gridsize < 10:
        raise ValueError("--hexbin-gridsize deve ser pelo menos 10.")


def main() -> int:
    """Processa o fluxo completo de análise espacial."""
    args = parse_args()

    try:
        validate_args(args)

        transactions_path = resolve_path(args.transactions)
        indicators_path = resolve_path(args.sector_indicators)
        census_path = resolve_path(args.census)
        mesh_path = resolve_path(args.sector_mesh)
        processed_dir = resolve_path(args.processed_dir)
        maps_dir = resolve_path(args.maps_dir)
        figures_dir = resolve_path(args.figures_dir)
        metadata_path = resolve_path(args.metadata)

        sectors = read_sector_mesh(mesh_path, str(args.municipality))
        indicators = load_sector_indicators(indicators_path)
        transactions = load_transactions(transactions_path)
        census = load_census(census_path)

        years = sorted(
            indicators["ano_analise"].dropna().astype(int).unique().tolist()
        )
        if not years:
            raise ValueError("Indicadores setoriais não contêm anos válidos.")

        incomplete_years = determine_incomplete_years(indicators)
        local_year = latest_complete_year(years, incomplete_years)

        value_vmin, value_vmax = common_value_scale(
            indicators,
            args.min_transactions,
        )
        volume_vmin, volume_vmax = common_volume_scale(indicators)

        generated_maps: list[str] = []
        global_rows: list[dict[str, Any]] = []
        local_frames: list[pd.DataFrame] = []

        for year in years:
            geodata = build_year_geodata(sectors, indicators, year)

            value_map = maps_dir / f"sector_value_m2_{year}.png"
            plot_sector_metric(
                geodata=geodata,
                column=VALUE_METRIC,
                title=(
                    f"Valor real mediano da base tributária por m² — {year}"
                ),
                legend_label="R$/m² reais",
                output_path=value_map,
                vmin=value_vmin,
                vmax=value_vmax,
                minimum_transactions=args.min_transactions,
            )
            generated_maps.append(str(value_map.relative_to(PROJECT_ROOT)))

            volume_map = maps_dir / f"sector_transactions_{year}.png"
            plot_sector_metric(
                geodata=geodata,
                column=VOLUME_METRIC,
                title=f"Número de transações por setor — {year}",
                legend_label="transações",
                output_path=volume_map,
                vmin=volume_vmin,
                vmax=volume_vmax,
            )
            generated_maps.append(str(volume_map.relative_to(PROJECT_ROOT)))

            heatmap = maps_dir / f"transaction_heatmap_{year}.png"
            plot_transaction_heatmap(
                transactions=transactions,
                sectors=sectors,
                year=year,
                output_path=heatmap,
                gridsize=args.hexbin_gridsize,
            )
            if heatmap.exists():
                generated_maps.append(str(heatmap.relative_to(PROJECT_ROOT)))

            incomplete = year in incomplete_years

            value_frame = prepare_value_moran_frame(
                sectors,
                indicators,
                year,
                args.min_transactions,
            )
            global_rows.append(
                calculate_global_moran(
                    geodata=value_frame,
                    metric=VALUE_METRIC,
                    year=year,
                    permutations=args.permutations,
                    seed=args.seed + year,
                    min_transactions=args.min_transactions,
                    incomplete_year=incomplete,
                )
            )

            volume_frame = prepare_volume_moran_frame(
                sectors,
                indicators,
                year,
            )
            global_rows.append(
                calculate_global_moran(
                    geodata=volume_frame,
                    metric="log_transacoes",
                    year=year,
                    permutations=args.permutations,
                    seed=args.seed + 10_000 + year,
                    min_transactions=None,
                    incomplete_year=incomplete,
                )
            )

            # Processa Local Moran para todos os anos para permitir análise
            # longitudinal posterior, mesmo que apenas o último ano completo
            # receba o mapa LISA principal nesta etapa.
            local = calculate_local_moran(
                geodata=value_frame,
                year=year,
                permutations=args.permutations,
                significance=args.significance,
                seed=args.seed + 20_000 + year,
            )
            local_frames.append(
                pd.DataFrame(
                    local.drop(columns=local.geometry.name)
                ).reset_index(drop=True)
            )

            if year == local_year:
                lisa_map = maps_dir / f"local_moran_value_{year}.png"
                plot_local_moran(
                    sectors=sectors,
                    local=local,
                    year=year,
                    output_path=lisa_map,
                )
                generated_maps.append(
                    str(lisa_map.relative_to(PROJECT_ROOT))
                )

        census_maps = plot_census_maps(
            sectors=sectors,
            census=census,
            maps_dir=maps_dir,
        )
        generated_maps.extend(census_maps)

        moran_table = pd.DataFrame(global_rows).sort_values(
            ["metric", "ano_analise"]
        )
        concentration = build_concentration_table(
            sectors,
            indicators,
        )
        local_table = pd.concat(
            local_frames,
            ignore_index=True,
            sort=False,
        )

        processed_dir.mkdir(parents=True, exist_ok=True)

        moran_path = processed_dir / "spatial_global_moran.csv"
        local_path = processed_dir / "spatial_local_moran_value.parquet"
        concentration_path = processed_dir / "spatial_concentration.csv"

        moran_table.to_csv(moran_path, index=False, encoding="utf-8")
        local_table.to_parquet(local_path, index=False)
        concentration.to_csv(
            concentration_path,
            index=False,
            encoding="utf-8",
        )

        moran_figure = figures_dir / "global_moran_by_year.png"
        plot_global_moran_series(
            moran_table,
            moran_figure,
        )

        metadata = build_metadata(
            years=years,
            incomplete_years=incomplete_years,
            min_transactions=args.min_transactions,
            permutations=args.permutations,
            significance=args.significance,
            local_year=local_year,
            moran_table=moran_table,
            concentration=concentration,
            generated_maps=generated_maps,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print("\n[ok] Análise espacial concluída")
        print(f"     anos analisados: {len(years)}")
        print(f"     ano LISA principal: {local_year}")
        print(f"     mapas gerados: {len(generated_maps)}")
        print(f"     {moran_path.relative_to(PROJECT_ROOT)}")
        print(f"     {local_path.relative_to(PROJECT_ROOT)}")
        print(f"     {concentration_path.relative_to(PROJECT_ROOT)}")
        print(f"     {moran_figure.relative_to(PROJECT_ROOT)}")
        print(f"     {metadata_path.relative_to(PROJECT_ROOT)}")

        print("\nMoran global:")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(moran_table.to_string(index=False))

        print("\nConcentração espacial:")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(concentration.to_string(index=False))

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
