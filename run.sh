#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  —  Generate + validate STAC catalog
# Usage:
#   ./run.sh               full index (all out_* folders, all variables)
#   ./run.sh --limit 5     test mode  (5 files per collection)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/users/nfarabul/data-compression/netCDF_files"
OUT_DIR="$SCRIPT_DIR/stac_out"
CATALOG_PY="$SCRIPT_DIR/generate_catalog.py"
APP_PY="$SCRIPT_DIR/app.py"

LIMIT_ARG=""
if [[ "${1:-}" == "--limit" && -n "${2:-}" ]]; then
    LIMIT_ARG="--limit $2"
    echo "  [test mode] limiting to $2 items per collection"
fi

# ── venv python/pip ───────────────────────────────────────────────────────────
VENV="$SCRIPT_DIR/venv"
if [[ -f "$VENV/bin/python" ]]; then
    PYTHON="$VENV/bin/python"
    PIP="$VENV/bin/pip"
else
    PYTHON="$(which python)"
    PIP="$(which pip)"
fi

echo "══════════════════════════════════════════════"
echo "  DYAMOND STAC Catalog Generator"
echo "  python  : $PYTHON"
echo "  data    : $DATA_DIR"
echo "  output  : $OUT_DIR"
echo "══════════════════════════════════════════════"

# ── 1. preflight checks ───────────────────────────────────────────────────────
echo ""
echo "[1/5] Checking inputs..."

[[ -f "$CATALOG_PY" ]] || { echo "[ERROR] generate_catalog.py not found"; exit 1; }
[[ -f "$APP_PY"     ]] || { echo "[ERROR] app.py not found"; exit 1; }
[[ -d "$DATA_DIR"   ]] || { echo "[ERROR] DATA_DIR not found: $DATA_DIR"; exit 1; }

# direct .nc files
NC_DIRECT=$(find "$DATA_DIR" -maxdepth 1 -name "*.nc" | wc -l)
echo "  direct .nc files : $NC_DIRECT"

# out_* folders directly inside DATA_DIR
OUT_DIRS=$(find "$DATA_DIR" -maxdepth 1 -name "out_*" -type d | wc -l)
echo "  out_* folders    : $OUT_DIRS"
for d in "$DATA_DIR"/out_*/; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")
    count=$(find "$d" -maxdepth 1 -name "*.nc" | wc -l)
    vars=$(find "$d" -maxdepth 1 -name "remap_*.nc" \
           | sed 's/.*remap_//' \
           | sed 's/_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T.*//' \
           | sort -u | tr '\n' ' ')
    echo "    $name : $count files  vars=[$vars]"
done

# ── 2. dependencies ───────────────────────────────────────────────────────────
echo ""
echo "[2/5] Checking Python dependencies..."
$PIP install --quiet pystac xarray netCDF4 shapely pyproj stac_valid python-dateutil

for module in pystac xarray dateutil shapely; do
    if ! $PYTHON -c "import $module" 2>/dev/null; then
        echo "[ERROR] module '$module' not importable"
        echo "        $PIP install --force-reinstall $module"
        exit 1
    fi
done
echo "  [ok] all dependencies ready"

# ── 3. clean old catalog ──────────────────────────────────────────────────────
echo ""
echo "[3/5] Cleaning old stac_out/ ..."
[[ -d "$OUT_DIR" ]] && rm -rf "$OUT_DIR" && echo "  [ok] removed"

# ── 4. generate catalog ───────────────────────────────────────────────────────
echo ""
echo "[4/5] Generating catalog..."
echo "      (this may take a long time for the full dataset)"
$PYTHON "$CATALOG_PY" $LIMIT_ARG

# ── 5. validate ───────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Validating catalog..."
$PYTHON - "$OUT_DIR/catalog.json" << 'PYVAL'
import sys, json, pathlib
try:
    from stac_valid import validate
    jsons = [
        f for f in pathlib.Path(sys.argv[1]).parent.rglob("*.json")
        if ".ipynb_checkpoints" not in str(f)
    ]
    errors = []
    for f in jsons:
        try:
            if not validate(str(f)):
                errors.append(str(f))
        except Exception as e:
            errors.append(f"{f}: {e}")
    if errors:
        print(f"  [warn] {len(errors)} file(s) failed validation (first 5):")
        for e in errors[:5]:
            print(f"    {e}")
    else:
        print(f"  [ok] all {len(jsons)} JSON file(s) valid")
except ImportError:
    with open(sys.argv[1]) as fh:
        json.load(fh)
    print("  [ok] catalog.json parseable (stac_valid not installed)")
PYVAL

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  Done. Collections:"
find "$OUT_DIR" -name "collection.json" \
    | grep -v ".ipynb_checkpoints" \
    | sort \
    | while read f; do
        cid=$($PYTHON -c "import json; d=json.load(open('$f')); print(d['id'])" 2>/dev/null)
        n=$(find "$(dirname "$f")" -name "*.json" ! -name "collection.json" \
            | grep -v ".ipynb_checkpoints" | wc -l)
        vars=$($PYTHON -c \
            "import json; d=json.load(open('$f')); print(' '.join(d.get('variables',[])))" \
            2>/dev/null)
        echo "    $cid  ($n items)  vars=[$vars]"
    done

echo ""
echo "  To restart the API:"
echo "    kill \$(cat $SCRIPT_DIR/api.pid) 2>/dev/null || true"
echo "    nohup $PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8000 \\"
echo "        > $SCRIPT_DIR/api.log 2>&1 &"
echo "    echo \$! > $SCRIPT_DIR/api.pid"
echo ""
echo "  On your laptop:"
echo "    ssh -N -L 8000:\$(hostname):8000 santis"
echo "══════════════════════════════════════════════"

hostname > "$SCRIPT_DIR/api.node"