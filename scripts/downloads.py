#!/usr/bin/env python3
"""Baixa as bases brutas usadas na análise imobiliária de Porto Alegre.

Fontes cobertas:
- ITBI da Prefeitura de Porto Alegre (2020 em diante, via CKAN);
- limites oficiais de bairros, Regiões de Planejamento e Regiões do OP (PMPA);
- Censo 2022/IBGE por setor censitário: básico, demografia e renda do responsável;
- malha de setores censitários do RS (Censo 2022);
- CNEFE 2022 de Porto Alegre, com endereços/coordenadas para geocodificação;
- IPCA mensal (BCB/SGS 433) para deflacionar valores ao longo do tempo.

Os arquivos são mantidos em raw/ sem transformação. O script também grava
raw/manifest.json com URL, tamanho e SHA-256 para reprodutibilidade.

Uso:
    python scripts/downloads.py
    python scripts/downloads.py --groups itbi geo
    python scripts/downloads.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

USER_AGENT = (
    "PropertyValuation/1.0 "
    "(academic real-estate market research; Porto Alegre, Brazil)"
)
CHUNK_SIZE = 1024 * 1024
RETRIES = 3
TIMEOUT = 180

CKAN_BASE = "https://dadosabertos.poa.br"
IBGE_CENSO_BASE = (
    "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/"
    "Agregados_por_Setores_Censitarios/"
)
IBGE_RENDA_BASE = (
    "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/"
    "Agregados_por_Setores_Censitarios_Rendimento_do_Responsavel/"
)

CURRENT_FALLBACKS = {
    "censo_basico": (
        IBGE_CENSO_BASE
        + "Agregados_por_Setor_csv/Agregados_por_setores_basico_BR_20260520.zip"
    ),
    "censo_demografia": (
        IBGE_CENSO_BASE
        + "Agregados_por_Setor_csv/Agregados_por_setores_demografia_BR.zip"
    ),
    "censo_bairros_basico": (
        IBGE_CENSO_BASE
        + "Agregados_por_Bairro_csv/Agregados_por_bairros_basico_BR_20260520.zip"
    ),
    "censo_bairros_demografia": (
        IBGE_CENSO_BASE
        + "Agregados_por_Bairro_csv/Agregados_por_bairros_demografia_BR.zip"
    ),
    "censo_dicionario": (
        IBGE_CENSO_BASE
        + "dicionario_de_dados_agregados_por_setores_censitarios_20260520.xlsx"
    ),
    "renda_setores": (
        IBGE_RENDA_BASE
        + "Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip"
    ),
    "renda_bairros": (
        IBGE_RENDA_BASE
        + "Agregados_por_bairros_renda_responsavel_BR_20260508_csv.zip"
    ),
    "renda_dicionario": (
        IBGE_RENDA_BASE + "dicionario_de_dados_renda_responsavel_20260508.xlsx"
    ),
}

STATIC_SOURCES = {
    "geo": [
        (
            "ibge_setores_rs",
            "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/"
            "Agregados_por_Setores_Censitarios/malha_com_atributos/setores/"
            "shp/UF/RS/RS_setores_CD2022.zip",
            RAW_DIR / "ibge" / "censo2022" / "malhas" / "RS_setores_CD2022.zip",
        ),
    ],
    "cnefe": [
        (
            "cnefe_porto_alegre",
            "https://ftp.ibge.gov.br/Cadastro_Nacional_de_Enderecos_para_Fins_Estatisticos/"
            "Censo_Demografico_2022/Arquivos_CNEFE/CSV/Municipio/43_RS/"
            "4314902_PORTO_ALEGRE.zip",
            RAW_DIR / "ibge" / "cnefe2022" / "4314902_PORTO_ALEGRE.zip",
        ),
        (
            "cnefe_dicionario",
            "https://ftp.ibge.gov.br/Cadastro_Nacional_de_Enderecos_para_Fins_Estatisticos/"
            "Censo_Demografico_2022/Arquivos_CNEFE/CSV/Dicionario_CNEFE_Censo_2022.xls",
            RAW_DIR / "ibge" / "cnefe2022" / "Dicionario_CNEFE_Censo_2022.xls",
        ),
    ],
}

CKAN_DATASETS = {
    "itbi": [
        {
            "slug": "itbi",
            "target": RAW_DIR / "porto_alegre" / "itbi",
            "formats": {"csv", "pdf"},
        }
    ],
    "geo": [
        {
            "slug": "bairros-lc-12-112-16",
            "target": RAW_DIR / "porto_alegre" / "geometria" / "bairros",
            "formats": {"zip", "pdf"},
        },
        {
            "slug": "regioes-de-planejamento-rp",
            "target": RAW_DIR
            / "porto_alegre"
            / "geometria"
            / "regioes_planejamento",
            "formats": {"zip"},
        },
        {
            "slug": "regioes-do-orcamento-participativo-rop",
            "target": RAW_DIR / "porto_alegre" / "geometria" / "regioes_op",
            "formats": {"zip", "pdf"},
        },
    ],
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)
                break


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_bytes(url: str, timeout: int = TIMEOUT) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == RETRIES:
                break
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, force: bool = False) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[skip] {destination.relative_to(PROJECT_ROOT)}")
        return {
            "status": "existing",
            "url": url,
            "path": str(destination.relative_to(PROJECT_ROOT)),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "checked_at": utc_now_iso(),
        }

    temp = destination.with_suffix(destination.suffix + ".part")
    if temp.exists():
        temp.unlink()

    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            print(f"[download] {url}")
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urlopen(request, timeout=TIMEOUT) as response, temp.open("wb") as output:
                total = int(response.headers.get("Content-Length") or 0)
                received = 0
                next_report = 10
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    if total:
                        pct = int(received * 100 / total)
                        if pct >= next_report:
                            print(f"           {pct:3d}%", end="\r", flush=True)
                            next_report += 10
            if total:
                print("           100%")
            temp.replace(destination)
            return {
                "status": "downloaded",
                "url": url,
                "path": str(destination.relative_to(PROJECT_ROOT)),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "downloaded_at": utc_now_iso(),
            }
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if temp.exists():
                temp.unlink()
            if attempt < RETRIES:
                print(f"[retry {attempt}/{RETRIES}] {exc}", file=sys.stderr)
                time.sleep(2**attempt)

    assert last_error is not None
    raise last_error


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "arquivo"


def normalize_resource_format(resource: dict) -> str:
    values = [
        str(resource.get("format") or "").lower(),
        str(resource.get("mimetype") or "").lower(),
        str(resource.get("url") or "").lower(),
    ]
    joined = " ".join(values)
    for fmt in ("csv", "zip", "pdf", "xlsx", "xls", "geojson", "json"):
        if fmt in joined:
            return fmt
    return (resource.get("format") or "").lower().strip(".")


def resource_filename(resource: dict) -> str:
    url = resource.get("url") or ""
    candidate = unquote(Path(urlparse(url).path).name)
    if candidate and "." in candidate:
        return candidate

    name = safe_name(resource.get("name") or resource.get("id") or "recurso")
    fmt = normalize_resource_format(resource)
    return f"{name}.{fmt}" if fmt else name


def download_ckan_dataset(
    slug: str,
    target_dir: Path,
    formats: set[str],
    *,
    force: bool,
) -> list[dict]:
    api_url = f"{CKAN_BASE}/api/3/action/package_show?{urlencode({'id': slug})}"
    payload = json.loads(request_bytes(api_url).decode("utf-8"))
    if not payload.get("success"):
        raise RuntimeError(f"CKAN retornou success=false para {slug}")

    metadata_dir = RAW_DIR / "_metadata" / "ckan"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{slug}.json"
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results: list[dict] = []
    resources = payload["result"].get("resources", [])
    for resource in resources:
        fmt = normalize_resource_format(resource)
        if fmt not in formats:
            continue
        url = resource.get("url")
        if not url:
            continue
        url = urljoin(CKAN_BASE, url)
        destination = target_dir / resource_filename(resource)
        item = download_file(url, destination, force=force)
        item.update(
            {
                "source": f"CKAN Porto Alegre: {slug}",
                "resource_name": resource.get("name"),
                "resource_id": resource.get("id"),
            }
        )
        results.append(item)
    return results


def directory_files(url: str) -> list[str]:
    html = request_bytes(url).decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(html)
    files: list[str] = []
    for href in parser.hrefs:
        name = unquote(Path(urlparse(href).path).name)
        if name:
            files.append(name)
    return files


def latest_matching_file(
    directory_url: str,
    pattern: str,
    fallback_url: str,
) -> str:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    try:
        matches = [name for name in directory_files(directory_url) if regex.fullmatch(name)]
    except Exception as exc:
        print(
            f"[warn] Não foi possível listar {directory_url}: {exc}. Usando fallback.",
            file=sys.stderr,
        )
        return fallback_url

    if not matches:
        print(
            f"[warn] Nenhum arquivo compatível em {directory_url}. Usando fallback.",
            file=sys.stderr,
        )
        return fallback_url

    # Nomes datados no IBGE usam AAAAMMDD; ordem lexicográfica escolhe a versão mais nova.
    return urljoin(directory_url, sorted(matches)[-1])


def download_censo(*, force: bool) -> list[dict]:
    target = RAW_DIR / "ibge" / "censo2022" / "agregados_setor"
    target.mkdir(parents=True, exist_ok=True)

    setor_csv_dir = IBGE_CENSO_BASE + "Agregados_por_Setor_csv/"
    specs = [
        (
            "censo_basico",
            setor_csv_dir,
            r"Agregados_por_setores_basico_BR(?:_\d{8})?\.zip",
            CURRENT_FALLBACKS["censo_basico"],
        ),
        (
            "censo_demografia",
            setor_csv_dir,
            r"Agregados_por_setores_demografia_BR(?:_\d{8})?\.zip",
            CURRENT_FALLBACKS["censo_demografia"],
        ),
        (
            "censo_dicionario",
            IBGE_CENSO_BASE,
            r"dicionario_de_dados_agregados_por_setores_censitarios(?:_\d{8})?\.xlsx",
            CURRENT_FALLBACKS["censo_dicionario"],
        ),
    ]

    results: list[dict] = []
    for label, directory_url, pattern, fallback in specs:
        url = latest_matching_file(directory_url, pattern, fallback)
        destination = target / unquote(Path(urlparse(url).path).name)
        item = download_file(url, destination, force=force)
        item["source"] = f"IBGE Censo 2022: {label}"
        results.append(item)

    # Agregados oficiais por bairro: evitam reconstruir médias de renda a partir
    # de setores (o denominador exato da média não é divulgado no arquivo setorial).
    bairro_target = RAW_DIR / "ibge" / "censo2022" / "agregados_bairro"
    bairro_target.mkdir(parents=True, exist_ok=True)
    bairro_csv_dir = IBGE_CENSO_BASE + "Agregados_por_Bairro_csv/"
    bairro_specs = [
        (
            "censo_bairros_basico",
            r"Agregados_por_bairros_basico_BR(?:_\d{8})?\.zip",
            CURRENT_FALLBACKS["censo_bairros_basico"],
        ),
        (
            "censo_bairros_demografia",
            r"Agregados_por_bairros_demografia_BR(?:_\d{8})?\.zip",
            CURRENT_FALLBACKS["censo_bairros_demografia"],
        ),
    ]
    for label, pattern, fallback in bairro_specs:
        url = latest_matching_file(bairro_csv_dir, pattern, fallback)
        destination = bairro_target / unquote(Path(urlparse(url).path).name)
        item = download_file(url, destination, force=force)
        item["source"] = f"IBGE Censo 2022: {label}"
        results.append(item)

    renda_target = RAW_DIR / "ibge" / "censo2022" / "renda_responsavel"
    renda_target.mkdir(parents=True, exist_ok=True)
    renda_specs = [
        (
            "renda_setores",
            r"Agregados_por_setores_renda_responsavel_BR_\d{8}_csv\.zip",
            CURRENT_FALLBACKS["renda_setores"],
        ),
        (
            "renda_dicionario",
            r"dicionario_de_dados_renda_responsavel_\d{8}\.xlsx",
            CURRENT_FALLBACKS["renda_dicionario"],
        ),
    ]
    for label, pattern, fallback in renda_specs:
        url = latest_matching_file(IBGE_RENDA_BASE, pattern, fallback)
        destination = renda_target / unquote(Path(urlparse(url).path).name)
        item = download_file(url, destination, force=force)
        item["source"] = f"IBGE Censo 2022: {label}"
        results.append(item)

    renda_bairro_target = RAW_DIR / "ibge" / "censo2022" / "renda_responsavel_bairro"
    renda_bairro_target.mkdir(parents=True, exist_ok=True)
    url = latest_matching_file(
        IBGE_RENDA_BASE,
        r"Agregados_por_bairros_renda_responsavel_BR_\d{8}_csv\.zip",
        CURRENT_FALLBACKS["renda_bairros"],
    )
    destination = renda_bairro_target / unquote(Path(urlparse(url).path).name)
    item = download_file(url, destination, force=force)
    item["source"] = "IBGE Censo 2022: renda_bairros"
    results.append(item)

    return results


def download_static_group(group: str, *, force: bool) -> list[dict]:
    results: list[dict] = []
    for label, url, destination in STATIC_SOURCES.get(group, []):
        item = download_file(url, destination, force=force)
        item["source"] = label
        results.append(item)
    return results


def download_ipca(*, force: bool) -> list[dict]:
    start = "01/01/2020"
    end = date.today().strftime("%d/%m/%Y")
    query = urlencode(
        {
            "formato": "json",
            "dataInicial": start,
            "dataFinal": end,
        }
    )
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?{query}"
    destination = RAW_DIR / "macro" / "ipca_mensal_sgs433_2020_ate_hoje.json"
    item = download_file(url, destination, force=force)
    item["source"] = "Banco Central do Brasil - SGS 433 (IPCA mensal)"
    return [item]


def write_manifest(entries: list[dict], errors: list[dict], groups: Iterable[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": utc_now_iso(),
        "project": "PropertyValuation",
        "scope": "Porto Alegre, RS",
        "groups": list(groups),
        "files": entries,
        "errors": errors,
    }
    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa dados brutos para a análise imobiliária de Porto Alegre."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["itbi", "geo", "censo", "cnefe", "macro"],
        default=["itbi", "geo", "censo", "cnefe", "macro"],
        help="Grupos a baixar. Por padrão, baixa todos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Baixa novamente mesmo quando o arquivo já existe em data/raw/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups: list[str] = list(dict.fromkeys(args.groups))
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    errors: list[dict] = []

    def run(label: str, func) -> None:
        try:
            entries.extend(func())
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"group": label, "error": message})
            print(f"[erro] {label}: {message}", file=sys.stderr)

    for group in groups:
        if group in CKAN_DATASETS:
            for dataset in CKAN_DATASETS[group]:
                run(
                    f"{group}:{dataset['slug']}",
                    lambda dataset=dataset: download_ckan_dataset(
                        dataset["slug"],
                        dataset["target"],
                        dataset["formats"],
                        force=args.force,
                    ),
                )

        if group == "censo":
            run("censo", lambda: download_censo(force=args.force))
        elif group == "cnefe":
            run("cnefe", lambda: download_static_group("cnefe", force=args.force))
        elif group == "geo":
            run("geo:ibge", lambda: download_static_group("geo", force=args.force))
        elif group == "macro":
            run("macro:ipca", lambda: download_ipca(force=args.force))

    write_manifest(entries, errors, groups)

    print(f"\nManifesto: {(RAW_DIR / 'manifest.json').relative_to(PROJECT_ROOT)}")
    print(f"Arquivos registrados: {len(entries)}")
    if errors:
        print(f"Falhas: {len(errors)} (veja raw/manifest.json)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
