#!/usr/bin/env python3
"""Integra transações ITBI georreferenciadas ao perfil socioeconômico do setor.

Entradas:
    data/processed/itbi_transactions_geocoded.parquet
    data/processed/census_sector_profile.parquet

Saídas:
    data/processed/itbi_transactions_enriched.parquet
    data/processed/itbi_census_join_summary.json

A ligação é feita por:
    cnefe_sector_code (CNEFE do endereço) == CD_SETOR (Censo 2022)

Nenhuma linha é removida. Registros sem geocodificação ou sem perfil censitário
permanecem na base e recebem flags de qualidade.

Uso:
    python scripts/05_join_itbi_census.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITBI = (
    PROJECT_ROOT / "data" / "processed" / "itbi_transactions_geocoded.parquet"
)
DEFAULT_CENSUS = (
    PROJECT_ROOT / "data" / "processed" / "census_sector_profile.parquet"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integra ITBI georreferenciado e Censo 2022 por setor censitário."
    )
    parser.add_argument("--itbi", type=Path, default=DEFAULT_ITBI)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    itbi_path = resolve(args.itbi)
    census_path = resolve(args.census)
    output_path = resolve(args.output)
    summary_path = resolve(args.summary)

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

        if "cnefe_sector_code" not in itbi.columns:
            raise ValueError("ITBI sem coluna cnefe_sector_code.")
        if "CD_SETOR" not in census.columns:
            raise ValueError("Censo sem coluna CD_SETOR.")

        itbi = itbi.copy()
        census = census.copy()

        itbi["sector_join_key"] = normalize_sector(itbi["cnefe_sector_code"])
        census["sector_join_key"] = normalize_sector(census["CD_SETOR"])

        if census["sector_join_key"].duplicated().any():
            raise ValueError("Perfil censitário contém CD_SETOR duplicado.")

        # Evita duplicar a chave original do Censo com um nome ambíguo.
        census_payload = census.rename(columns={"CD_SETOR": "census_sector_code"})

        enriched = itbi.merge(
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(output_path, index=False)

        total = len(enriched)
        geocoded = int(enriched["latitude"].notna().sum())
        census_matched = int(enriched["flag_census_matched"].sum())
        ready = int(enriched["flag_ready_for_spatial_analysis"].sum())

        summary = {
            "generated_at": utc_now_iso(),
            "transactions": total,
            "geocoded": geocoded,
            "geocoded_pct": round(100 * geocoded / total, 3) if total else None,
            "census_matched": census_matched,
            "census_matched_pct": (
                round(100 * census_matched / total, 3) if total else None
            ),
            "ready_for_spatial_analysis": ready,
            "ready_for_spatial_analysis_pct": (
                round(100 * ready / total, 3) if total else None
            ),
            "join": "cnefe_sector_code == CD_SETOR",
            "output": str(output_path.relative_to(PROJECT_ROOT)),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n[ok] ITBI + Censo integrados")
        print(f"     transações: {total:,}")
        print(f"     geocodificadas: {geocoded:,}")
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
