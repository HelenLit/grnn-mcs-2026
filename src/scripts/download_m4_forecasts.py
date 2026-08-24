"""Download the competitor point forecasts of the M4 competition.

Source: github.com/Mcompetitions/M4-methods, folder `Point Forecasts/` —
62 files: `Submission Info.xlsx` + 61 `submission-<id>.rar` archives
(~460 MB total), each archive holding one wide-format CSV of the method's
forecasts for all 100,000 series (row = series, columns = horizon steps).

Files land in `data/raw/m4/point_forecasts/` (read-only raw-data convention):
    point_forecasts/
        rar/          the downloaded .rar archives (kept for provenance)
        csv/          extracted submission CSVs (with --extract)

Usage:
    python -m src.scripts.download_m4_forecasts --list
    python -m src.scripts.download_m4_forecasts --only 118,245,Naive2,Com --extract
    python -m src.scripts.download_m4_forecasts --all --extract

Notes:
    - Stdlib only (urllib); no GitHub token needed (2 API calls per run).
    - Resumable: a file whose local size matches the GitHub-reported size is skipped.
    - Extraction tries `unar`, `unrar`, `7z`/`7zz`, then `bsdtar`. If none can
      read the archive: `brew install unar`.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

API_URL = "https://api.github.com/repos/Mcompetitions/M4-methods/contents/Point%20Forecasts"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "data" / "raw" / "m4" / "point_forecasts"

# Required submissions:
# 118 = Smyl (winner), 245 = Montero-Manso FFORMA (runner-up),
# Com = "Comb" benchmark bar, ARIMA/MLP = official benchmark submissions,
# Naive2 = OWA denominator.
DEFAULT_ONLY = "118,245,Com,ARIMA,MLP,Naive2"
REQUIRED_SUBMISSIONS = tuple(DEFAULT_ONLY.split(","))
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "results" / "m4"


def fetch_listing() -> list[dict]:
    req = urllib.request.Request(API_URL, headers={"User-Agent": "grnn-mcs-research"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        items = json.load(resp)
    return [it for it in items if it["type"] == "file"]


def _inventory_entries(items: list[dict]) -> list[dict]:
    return [
        {
            "name": item["name"],
            "size": item["size"],
            "sha": item["sha"],
            "download_url": item["download_url"],
        }
        for item in sorted(items, key=lambda entry: entry["name"].lower())
    ]


def write_inventory(items: list[dict], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    path = reports_dir / f"point_forecasts_inventory_{snapshot_date}.json"
    path.write_text(
        json.dumps(_inventory_entries(items), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_latest_inventory(reports_dir: Path) -> list[dict] | None:
    candidates = sorted(reports_dir.glob("point_forecasts_inventory_*.json"))
    if not candidates:
        return None
    with open(candidates[-1], "r", encoding="utf-8") as inventory_file:
        return json.load(inventory_file)


def wanted(items: list[dict], only: list[str] | None) -> list[dict]:
    if only is None:  # --all
        return items
    keys = {k.strip().lower() for k in only}
    picked = []
    for it in items:
        stem = it["name"].removeprefix("submission-").removesuffix(".rar").lower()
        if stem in keys or it["name"].lower() in keys:
            picked.append(it)
    found = {it["name"].removeprefix("submission-").removesuffix(".rar").lower() for it in picked}
    for miss in keys - found - {it["name"].lower() for it in picked}:
        print(f"  WARNING: no file matches '{miss}' (use --list to see ids)", file=sys.stderr)
    return picked


def expected_csv_path(item: dict, csv_dir: Path) -> Path:
    return csv_dir / f"{item['name'].removesuffix('.rar')}.csv"


def _archive_ok(item: dict, rar_dir: Path) -> bool:
    archive = rar_dir / item["name"]
    return archive.exists() and archive.stat().st_size == item["size"]


def _csv_ok(item: dict, csv_dir: Path) -> bool:
    return expected_csv_path(item, csv_dir).exists()


def _local_ok(item: dict, dest: Path) -> bool:
    return _archive_ok(item, dest / "rar") and _csv_ok(item, dest / "csv")


def download(item: dict, dest_dir: Path) -> Path:
    dest = dest_dir / item["name"]
    if dest.exists() and dest.stat().st_size == item["size"]:
        print(f"  skip (already complete): {item['name']}")
        return dest
    print(f"  downloading {item['name']} ({item['size'] / 1e6:.1f} MB) ...", flush=True)
    req = urllib.request.Request(item["download_url"], headers={"User-Agent": "grnn-mcs-research"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    if tmp.stat().st_size != item["size"]:
        tmp.unlink()
        raise RuntimeError(f"size mismatch for {item['name']}: "
                           f"got {tmp.stat().st_size}, expected {item['size']}")
    tmp.rename(dest)
    return dest


def _extract_cmd(tool: str, rar: Path, out_dir: Path) -> list[str]:
    return {
        "unar": [tool, "-force-overwrite", "-o", str(out_dir), str(rar)],
        "unrar": [tool, "x", "-o+", str(rar), str(out_dir) + "/"],
        "7z": [tool, "x", "-y", f"-o{out_dir}", str(rar)],
        "7zz": [tool, "x", "-y", f"-o{out_dir}", str(rar)],
        "bsdtar": [tool, "-x", "-f", str(rar), "-C", str(out_dir)],
    }[tool]


def extract(rar: Path, out_dir: Path) -> bool:
    if rar.suffix.lower() != ".rar":
        shutil.copy2(rar, out_dir / rar.name)  # e.g. Submission Info.xlsx
        return True
    for tool in ("unar", "unrar", "7z", "7zz", "bsdtar"):
        if shutil.which(tool) is None:
            continue
        proc = subprocess.run(_extract_cmd(tool, rar, out_dir),
                              capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"  extracted with {tool}: {rar.name}")
            return True
    print(f"  ERROR: could not extract {rar.name} — no working RAR tool.\n"
          f"         Install one with:  brew install unar", file=sys.stderr)
    return False


def _offline_items(
    dest: Path,
    reports_dir: Path,
    only: list[str] | None,
) -> list[dict] | None:
    cached = load_latest_inventory(reports_dir)
    if cached is None:
        return None
    picked = wanted(cached, only)
    if picked and all(_local_ok(item, dest) for item in picked):
        return picked
    return None


def sync_forecasts(
    dest: Path = DEFAULT_DEST,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    only: list[str] | None = None,
) -> dict:
    """Synchronize required M4 point forecasts and return a manifest report."""
    only = list(REQUIRED_SUBMISSIONS) if only is None else only
    inventory_path: Path | None = None
    offline = False

    try:
        items = fetch_listing()
        inventory_path = write_inventory(items, reports_dir)
        picked = wanted(items, only)
    except Exception as error:
        picked = _offline_items(dest, reports_dir, only)
        if picked is None:
            raise RuntimeError(
                "GitHub inventory unavailable and cached local forecast data "
                "could not be verified"
            ) from error
        offline = True

    rar_dir = dest / "rar"
    csv_dir = dest / "csv"
    rar_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    bytes_transferred = 0
    statuses = []
    for item in picked:
        archive = rar_dir / item["name"]
        had_archive = archive.exists()
        archive_matches = _archive_ok(item, rar_dir)
        csv_matches = _csv_ok(item, csv_dir)
        archive_changed = False

        if archive_matches and csv_matches:
            statuses.append({"file": item["name"], "status": "ok"})
            continue

        if offline:
            raise RuntimeError(
                f"{item['name']} is incomplete locally and cannot be refreshed offline"
            )

        status = "extracted"
        if not archive_matches:
            status = "re-downloaded" if had_archive else "downloaded"
            download(item, rar_dir)
            bytes_transferred += int(item["size"])
            archive_changed = True

        if archive_changed or not _csv_ok(item, csv_dir):
            if not extract(archive, csv_dir):
                raise RuntimeError(f"Could not extract {archive}")

        statuses.append({"file": item["name"], "status": status})

    if statuses and all(row["status"] == "ok" for row in statuses):
        message = "all prerequisite submission data is up to date"
        if offline:
            message += " (verified against cached inventory, offline)"
        print(message)

    return {
        "offline": offline,
        "inventory_path": str(inventory_path) if inventory_path else None,
        "required": only,
        "bytes_transferred": bytes_transferred,
        "files": statuses,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true",
                       help="download all 62 files (~460 MB)")
    group.add_argument("--only", default=DEFAULT_ONLY,
                       help="comma-separated ids, e.g. '118,245,Naive2,Com' "
                            f"(default: {DEFAULT_ONLY})")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"destination directory (default: {DEFAULT_DEST})")
    ap.add_argument("--extract", action="store_true",
                    help="extract downloaded .rar archives into <dest>/csv/")
    ap.add_argument("--list", action="store_true",
                    help="list available files and exit (no download)")
    ap.add_argument("--sync", action="store_true",
                    help="verify/download/extract required submissions")
    ap.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR,
                    help=f"sync inventory directory (default: {DEFAULT_REPORTS_DIR})")
    args = ap.parse_args()

    if args.sync:
        report = sync_forecasts(
            dest=args.dest,
            reports_dir=args.reports_dir,
            only=None if args.all else args.only.split(","),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    items = fetch_listing()
    if args.list:
        total = sum(it["size"] for it in items)
        for it in items:
            print(f"{it['size'] / 1e6:8.1f} MB  {it['name']}")
        print(f"{total / 1e6:8.1f} MB  TOTAL ({len(items)} files)")
        return 0

    picked = wanted(items, None if args.all else args.only.split(","))
    if not picked:
        print("Nothing matched — use --list to see available ids.", file=sys.stderr)
        return 1

    rar_dir = args.dest / "rar"
    rar_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(picked)} file(s) "
          f"({sum(it['size'] for it in picked) / 1e6:.1f} MB) -> {rar_dir}")
    archives = [download(it, rar_dir) for it in picked]

    if args.extract:
        csv_dir = args.dest / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        failures = sum(not extract(a, csv_dir) for a in archives)
        if failures:
            return 1
        print(f"Extracted to {csv_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
