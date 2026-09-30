#!/usr/bin/env python3
"""Processa gráficos e um relatório HTML consolidado do PropertyValuation.

Entradas:
    outputs/tables/market_indicators_city_year.csv
    outputs/tables/market_indicators_city_month.csv
    outputs/tables/market_indicators_city_ytd_comparable.csv
    outputs/tables/market_indicators_neighborhood_year.csv
    outputs/tables/market_indicators_op_region_year.csv
    outputs/tables/market_indicators_sector_year.csv
    outputs/maps/*.png

Saídas:
    outputs/figures/*.png
    outputs/property_valuation_report.html

Metodologia:
- processa indicadores consolidados das etapas anteriores;
- analisa séries anuais, mensais, YTD, bairros, regiões e setores;
- processa gráficos com escalas e filtros explícitos;
- analisa 2026 como ano parcial quando marcado na etapa 08;
- incorpora os mapas produzidos pela etapa 09 quando disponíveis;
- documenta objetivo, fontes, limpeza, variáveis e outputs do projeto.

Uso:
    python scripts/10_generate_report.py

Dependências:
    pandas
    numpy
    matplotlib
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_MAPS_DIR = PROJECT_ROOT / "outputs" / "maps"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "property_valuation_report.html"


def utc_now_iso() -> str:
    """Processa o instante atual em UTC para o rodapé do relatório."""
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path) -> Path:
    """Processa caminhos relativos sempre a partir da raiz do projeto."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_required(path: Path, required: Iterable[str]) -> pd.DataFrame:
    """Processa um CSV e analisa seu contrato mínimo."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name}: colunas obrigatórias ausentes: {sorted(missing)}"
        )
    return frame


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Processa a persistência padronizada de um gráfico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_city_transactions(city: pd.DataFrame, output: Path) -> Path:
    """Processa a série anual do número de transações."""
    fig, ax = plt.subplots(figsize=(9, 5))
    years = city["ano_analise"].astype(int)
    bars = ax.bar(years.astype(str), city["n_transacoes"])

    # Analisa anos incompletos e os diferencia visualmente por hachura.
    incomplete = city["flag_ano_incompleto"].fillna(False).astype(bool)
    for bar, is_incomplete in zip(bars, incomplete, strict=False):
        if is_incomplete:
            bar.set_hatch("//")
            bar.set_alpha(0.65)

    ax.set_title("Transações da amostra de mercado por ano")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Transações")
    ax.grid(axis="y", alpha=0.2)
    return save_figure(fig, output)


def plot_city_value_m2(city: pd.DataFrame, output: Path) -> Path:
    """Processa a evolução do valor real mediano da base tributária por m²."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        city["ano_analise"],
        city["valor_m2_mediano_real"],
        marker="o",
        linewidth=2,
    )
    ax.set_title("Valor real mediano da base tributária por m²")
    ax.set_xlabel("Ano")
    ax.set_ylabel("R$/m² reais")
    ax.grid(alpha=0.2)
    return save_figure(fig, output)


def plot_monthly_series(
    monthly: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output: Path,
) -> Path:
    """Processa uma série mensal contínua para analisar sazonalidade e tendência."""
    frame = monthly.copy()
    frame["date"] = pd.to_datetime(frame["periodo"] + "-01", errors="coerce")
    frame = frame.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(frame["date"], frame[value_column], linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Mês")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    return save_figure(fig, output)


def plot_ytd_comparison(
    ytd: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output: Path,
) -> Path:
    """Processa comparação YTD em janelas temporais equivalentes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ytd["ano_analise"].astype(str), ytd[value_column])
    cutoff = int(ytd["mes_limite_comparavel"].dropna().max())
    ax.set_title(f"{title} — janeiro a mês {cutoff:02d}")
    ax.set_xlabel("Ano")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    return save_figure(fig, output)


def plot_top_neighborhoods(
    neighborhoods: pd.DataFrame,
    year: int,
    metric: str,
    title: str,
    xlabel: str,
    output: Path,
    top_n: int = 15,
    min_transactions: int = 0,
) -> Path:
    """Processa ranking de bairros com filtro mínimo de observações."""
    frame = neighborhoods.loc[
        neighborhoods["ano_analise"].eq(year)
        & neighborhoods["n_transacoes"].ge(min_transactions)
    ].copy()
    frame = frame.nlargest(top_n, metric).sort_values(metric)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(frame["bairro_analise"], frame[metric])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.2)
    return save_figure(fig, output)


def plot_op_regions(
    op: pd.DataFrame,
    year: int,
    metric: str,
    title: str,
    xlabel: str,
    output: Path,
) -> Path:
    """Processa comparação entre Regiões do Orçamento Participativo."""
    frame = op.loc[op["ano_analise"].eq(year)].copy()
    frame = frame.sort_values(metric)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(frame["regiao_op"], frame[metric])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.2)
    return save_figure(fig, output)


def plot_sector_scatter(
    sectors: pd.DataFrame,
    year: int,
    output: Path,
) -> Path:
    """Analisa relação entre atividade e valor/m² entre setores."""
    frame = sectors.loc[
        sectors["ano_analise"].eq(year)
        & sectors["n_transacoes"].gt(0)
        & sectors["valor_m2_mediano_real"].gt(0)
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        frame["n_transacoes"],
        frame["valor_m2_mediano_real"],
        alpha=0.35,
        s=18,
    )
    ax.set_xscale("log")
    ax.set_title(f"Setores censitários: atividade × valor/m² — {year}")
    ax.set_xlabel("Transações no setor (escala log)")
    ax.set_ylabel("R$/m² reais — mediana")
    ax.grid(alpha=0.2)
    return save_figure(fig, output)


def image_data_uri(path: Path) -> str:
    """Processa uma imagem PNG como URI base64 para HTML autocontido."""
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def image_card(path: Path, title: str, caption: str = "") -> str:
    """Processa um card HTML de imagem quando o arquivo existe."""
    if not path.exists():
        return (
            '<div class="card missing"><h3>'
            + html.escape(title)
            + "</h3><p>Arquivo ainda não disponível: "
            + html.escape(str(path.relative_to(PROJECT_ROOT)))
            + "</p></div>"
        )

    return (
        '<figure class="card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<img src="{image_data_uri(path)}" alt="{html.escape(title)}">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def table_html(frame: pd.DataFrame, columns: list[str]) -> str:
    """Processa uma tabela compacta para o relatório."""
    view = frame[columns].copy()
    return view.to_html(index=False, border=0, classes="data-table")


def generate_report(
    city: pd.DataFrame,
    monthly: pd.DataFrame,
    ytd: pd.DataFrame,
    neighborhoods: pd.DataFrame,
    op: pd.DataFrame,
    sectors: pd.DataFrame,
    figures_dir: Path,
    maps_dir: Path,
    report_path: Path,
) -> None:
    """Processa gráficos e consolida o relatório HTML autocontido."""
    latest_full_year = int(
        city.loc[~city["flag_ano_incompleto"].fillna(False), "ano_analise"].max()
    )
    latest_year = int(city["ano_analise"].max())

    figures = {
        "transactions_year": plot_city_transactions(
            city,
            figures_dir / "market_transactions_year.png",
        ),
        "value_m2_year": plot_city_value_m2(
            city,
            figures_dir / "market_value_m2_year.png",
        ),
        "transactions_month": plot_monthly_series(
            monthly,
            "n_transacoes",
            "Transações mensais da amostra de mercado",
            "Transações",
            figures_dir / "market_transactions_month.png",
        ),
        "value_m2_month": plot_monthly_series(
            monthly,
            "valor_m2_mediano_real",
            "Valor real mediano da base tributária por m² — mensal",
            "R$/m² reais",
            figures_dir / "market_value_m2_month.png",
        ),
        "transactions_ytd": plot_ytd_comparison(
            ytd,
            "n_transacoes",
            "Transações YTD comparáveis",
            "Transações",
            figures_dir / "market_transactions_ytd.png",
        ),
        "value_m2_ytd": plot_ytd_comparison(
            ytd,
            "valor_m2_mediano_real",
            "Valor real mediano por m² — YTD comparável",
            "R$/m² reais",
            figures_dir / "market_value_m2_ytd.png",
        ),
        "neighborhood_volume": plot_top_neighborhoods(
            neighborhoods,
            latest_full_year,
            "n_transacoes",
            f"Bairros com mais transações — {latest_full_year}",
            "Transações",
            figures_dir / "top_neighborhoods_transactions.png",
        ),
        "neighborhood_value": plot_top_neighborhoods(
            neighborhoods,
            latest_full_year,
            "valor_m2_mediano_real",
            f"Maiores valores reais medianos por m² — {latest_full_year}",
            "R$/m² reais",
            figures_dir / "top_neighborhoods_value_m2.png",
            min_transactions=50,
        ),
        "op_volume": plot_op_regions(
            op,
            latest_full_year,
            "n_transacoes",
            f"Transações por Região do OP — {latest_full_year}",
            "Transações",
            figures_dir / "op_region_transactions.png",
        ),
        "op_value": plot_op_regions(
            op,
            latest_full_year,
            "valor_m2_mediano_real",
            f"Valor real mediano por m² por Região do OP — {latest_full_year}",
            "R$/m² reais",
            figures_dir / "op_region_value_m2.png",
        ),
        "sector_scatter": plot_sector_scatter(
            sectors,
            latest_full_year,
            figures_dir / "sector_activity_vs_value.png",
        ),
    }

    latest = city.loc[city["ano_analise"].eq(latest_year)].iloc[0]
    latest_full = city.loc[city["ano_analise"].eq(latest_full_year)].iloc[0]
    ytd_latest = ytd.loc[ytd["ano_analise"].eq(latest_year)].iloc[0]

    top_volume = (
        neighborhoods.loc[neighborhoods["ano_analise"].eq(latest_full_year)]
        .nlargest(10, "n_transacoes")
        [["bairro_analise", "n_transacoes", "valor_m2_mediano_real"]]
    )
    top_value = (
        neighborhoods.loc[
            neighborhoods["ano_analise"].eq(latest_full_year)
            & neighborhoods["n_transacoes"].ge(50)
        ]
        .nlargest(10, "valor_m2_mediano_real")
        [["bairro_analise", "n_transacoes", "valor_m2_mediano_real"]]
    )

    map_cards = []
    for year in sorted(city["ano_analise"].astype(int).unique()):
        map_cards.append(
            image_card(
                maps_dir / f"sector_value_m2_{year}.png",
                f"Valor real mediano por m² — setores — {year}",
                "Setores com amostra insuficiente podem aparecer sem valor.",
            )
        )
    map_cards.extend(
        [
            image_card(
                maps_dir / f"local_moran_value_{latest_full_year}.png",
                f"Clusters LISA do valor/m² — {latest_full_year}",
            ),
            image_card(
                maps_dir / "census_income_sector_2022.png",
                "Rendimento da pessoa responsável — Censo 2022",
            ),
            image_card(
                maps_dir / "census_age_sector_2022.png",
                "Idade média aproximada — Censo 2022",
            ),
            image_card(
                maps_dir / "census_household_size_sector_2022.png",
                "Média de moradores por domicílio — Censo 2022",
            ),
        ]
    )

    figure_cards = [
        image_card(figures["transactions_year"], "Transações anuais"),
        image_card(figures["value_m2_year"], "Valor real mediano por m²"),
        image_card(figures["transactions_ytd"], "Comparação YTD — transações"),
        image_card(figures["value_m2_ytd"], "Comparação YTD — valor/m²"),
        image_card(figures["transactions_month"], "Série mensal — transações"),
        image_card(figures["value_m2_month"], "Série mensal — valor/m²"),
        image_card(
            figures["neighborhood_volume"],
            "Bairros com maior atividade",
        ),
        image_card(
            figures["neighborhood_value"],
            "Bairros com maior valor/m²",
        ),
        image_card(figures["op_volume"], "Regiões do OP — atividade"),
        image_card(figures["op_value"], "Regiões do OP — valor/m²"),
        image_card(
            figures["sector_scatter"],
            "Setores — atividade versus valor/m²",
        ),
    ]

    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:0;color:#1f2937;background:#f7f8fa}
    header{padding:42px 7vw;background:#172554;color:white}
    main{max-width:1200px;margin:auto;padding:32px 5vw 70px}
    h1{margin:0 0 12px;font-size:34px} h2{margin-top:42px;color:#172554}
    h3{margin-top:0}.muted{color:#64748b}.warning{background:#fff7ed;border-left:4px solid #f97316;padding:14px}
    .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
    .kpi,.card{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(15,23,42,.05)}
    .kpi strong{display:block;font-size:25px;color:#172554;margin-top:6px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
    .card img{width:100%;height:auto}.missing{background:#f8fafc;color:#64748b}
    figcaption{font-size:12px;color:#64748b;margin-top:8px}
    .data-table{width:100%;border-collapse:collapse;background:white}
    .data-table th,.data-table td{padding:8px;border-bottom:1px solid #e5e7eb;text-align:right}
    .data-table th:first-child,.data-table td:first-child{text-align:left}
    code{background:#e2e8f0;padding:2px 5px;border-radius:4px}
    footer{margin-top:45px;color:#64748b;font-size:12px}
    """

    report = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PropertyValuation — relatório do projeto</title>
<style>{css}</style>
</head>
<body>
<header>
<h1>PropertyValuation</h1>
<p>Modelagem e análise do mercado imobiliário de Porto Alegre com ITBI, geocodificação, Censo 2022, IPCA e estatística espacial.</p>
</header>
<main>
<h2>Resumo executivo</h2>
<div class="kpis">
<div class="kpi">Último ano completo<strong>{latest_full_year}</strong></div>
<div class="kpi">Transações em {latest_full_year}<strong>{int(latest_full['n_transacoes']):,}</strong></div>
<div class="kpi">Valor mediano/m² em {latest_full_year}<strong>R$ {latest_full['valor_m2_mediano_real']:,.0f}</strong></div>
<div class="kpi">2026 — meses observados<strong>{int(latest['meses_observados'])}</strong></div>
<div class="kpi">2026 YTD — transações<strong>{int(ytd_latest['n_transacoes']):,}</strong></div>
<div class="kpi">2026 YTD — valor mediano/m²<strong>R$ {ytd_latest['valor_m2_mediano_real']:,.0f}</strong></div>
</div>

<p class="warning"><strong>Nota metodológica:</strong> <code>base_de_calculo</code> representa a base tributária publicada no ITBI. O projeto não a reclassifica como preço efetivo de venda. 2026 é tratado como período parcial; comparações com anos anteriores usam também uma janela YTD equivalente.</p>

<h2>Objetivo</h2>
<p>O projeto processa registros de ITBI para analisar a evolução temporal e espacial do mercado imobiliário de Porto Alegre, identificar diferenças entre bairros, regiões e setores censitários, integrar contexto socioeconômico do Censo 2022 e preparar bases auditáveis para indicadores, estatística espacial e modelagem imobiliária.</p>

<h2>Fontes de dados</h2>
<ul>
<li><strong>Prefeitura de Porto Alegre:</strong> registros de ITBI e geometrias municipais.</li>
<li><strong>IBGE — CNEFE 2022:</strong> referência de endereços para geocodificação.</li>
<li><strong>IBGE — Censo Demográfico 2022:</strong> agregados por setor e bairro, incluindo demografia e rendimento da pessoa responsável.</li>
<li><strong>IBGE — malha de setores censitários:</strong> associação espacial definitiva das transações.</li>
<li><strong>Banco Central do Brasil — SGS 433:</strong> IPCA mensal usado para correção monetária.</li>
</ul>

<h2>Processamento, limpeza e controle de qualidade</h2>
<ol>
<li>Processa downloads reproduzíveis e preserva a camada <code>data/raw</code>.</li>
<li>Processa ITBI por ano, normaliza datas, números, CEP, endereços, áreas e flags.</li>
<li>Analisa guias multiunidade sem distribuir arbitrariamente a base tributária entre unidades.</li>
<li>Processa geocodificação por CNEFE com métodos e escores de confiança.</li>
<li>Processa spatial join com a malha definitiva de setores e camadas municipais.</li>
<li>Processa variáveis demográficas e socioeconômicas do Censo 2022.</li>
<li>Processa correção monetária pelo IPCA e preserva valores nominais.</li>
<li>Analisa duplicidades, inconsistências e outliers; os registros são sinalizados, não apagados silenciosamente.</li>
<li>Processa amostras específicas para mercado, análise espacial, socioeconomia e modelos.</li>
</ol>

<h2>Variáveis principais</h2>
<ul>
<li><strong>Mercado:</strong> <code>base_de_calculo</code>, <code>base_de_calculo_real</code>, <code>valor_m2_base_real</code>, áreas, datas e número de unidades.</li>
<li><strong>Espaço:</strong> latitude, longitude, setor censitário, bairro, Região do OP e Região de Planejamento.</li>
<li><strong>Qualidade:</strong> flags de pagamento, cancelamento, duplicidade, geocodificação, IPCA, Censo e outliers.</li>
<li><strong>Censo:</strong> população, domicílios, densidade, idade aproximada, estrutura etária e rendimento médio da pessoa responsável com rendimento.</li>
</ul>

<h2>Indicadores e gráficos</h2>
<div class="grid">{''.join(figure_cards)}</div>

<h2>Bairros — {latest_full_year}</h2>
<h3>Maior número de transações</h3>
{table_html(top_volume, ['bairro_analise','n_transacoes','valor_m2_mediano_real'])}
<h3>Maior valor real mediano por m² — mínimo de 50 transações</h3>
{table_html(top_value, ['bairro_analise','n_transacoes','valor_m2_mediano_real'])}

<h2>Mapas e estatística espacial</h2>
<p>Os mapas abaixo são incorporados automaticamente quando a etapa <code>09_spatial_analysis.py</code> já foi executada no mesmo projeto. A análise utiliza setores censitários e inclui mapas de valor/m², volume de transações, Local Moran/LISA e contexto do Censo 2022.</p>
<div class="grid">{''.join(map_cards)}</div>

<h2>Outputs gerados pelo pipeline</h2>
<ul>
<li><code>data/interim/itbi_units_clean.parquet</code> e <code>itbi_transactions_clean.parquet</code>.</li>
<li><code>data/processed/itbi_transactions_geocoded.parquet</code> e <code>itbi_transactions_spatial.parquet</code>.</li>
<li><code>data/processed/census_sector_profile.parquet</code> e perfil por bairro.</li>
<li><code>data/processed/itbi_transactions_deflated.parquet</code>.</li>
<li><code>data/processed/itbi_transactions_qc.parquet</code> e amostra analítica.</li>
<li>Indicadores municipais, mensais, YTD, por bairro, Região do OP e setor.</li>
<li>Mapas, Moran global, Local Moran/LISA e medidas de concentração espacial.</li>
<li>Gráficos consolidados e este relatório HTML.</li>
</ul>

<h2>Leituras iniciais dos indicadores</h2>
<p>Entre os anos completos, a amostra municipal alcançou {int(city['n_transacoes'].max()):,} transações no maior volume anual observado. Em {latest_full_year}, o valor real mediano da base tributária por m² foi de R$ {latest_full['valor_m2_mediano_real']:,.0f}. Para {latest_year}, os indicadores anuais são parciais e a referência comparável é a tabela YTD.</p>

<footer>Relatório processado em {html.escape(utc_now_iso())}. Gerado por <code>scripts/10_generate_report.py</code>.</footer>
</main>
</body>
</html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Processa gráficos e relatório HTML do PropertyValuation."
    )
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Processa o fluxo completo de visualização e relatório."""
    args = parse_args()

    try:
        tables_dir = resolve_path(args.tables_dir)
        maps_dir = resolve_path(args.maps_dir)
        figures_dir = resolve_path(args.figures_dir)
        report_path = resolve_path(args.report)

        city = read_csv_required(
            tables_dir / "market_indicators_city_year.csv",
            ["ano_analise", "n_transacoes", "valor_m2_mediano_real"],
        )
        monthly = read_csv_required(
            tables_dir / "market_indicators_city_month.csv",
            ["periodo", "n_transacoes", "valor_m2_mediano_real"],
        )
        ytd = read_csv_required(
            tables_dir / "market_indicators_city_ytd_comparable.csv",
            ["ano_analise", "n_transacoes", "valor_m2_mediano_real"],
        )
        neighborhoods = read_csv_required(
            tables_dir / "market_indicators_neighborhood_year.csv",
            ["ano_analise", "bairro_analise", "n_transacoes", "valor_m2_mediano_real"],
        )
        op = read_csv_required(
            tables_dir / "market_indicators_op_region_year.csv",
            ["ano_analise", "regiao_op", "n_transacoes", "valor_m2_mediano_real"],
        )
        sectors = read_csv_required(
            tables_dir / "market_indicators_sector_year.csv",
            ["ano_analise", "sector_final_code", "n_transacoes", "valor_m2_mediano_real"],
        )

        generate_report(
            city=city,
            monthly=monthly,
            ytd=ytd,
            neighborhoods=neighborhoods,
            op=op,
            sectors=sectors,
            figures_dir=figures_dir,
            maps_dir=maps_dir,
            report_path=report_path,
        )

        print("\n[ok] Relatório consolidado concluído")
        print(f"     {report_path.relative_to(PROJECT_ROOT)}")
        print(f"     gráficos: {figures_dir.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
