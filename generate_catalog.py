"""
generate_catalog.py

Scans /users/nfarabul/data-compression/netCDF_files/ and builds a STAC catalog:

  Source A — direct .nc files in netCDF_files/:
    remap_<var>_<DATE>.nc         → collection "dyamond-direct-<var>"
    tigge_<domain>_<vars>_...nc   → collection "tigge-<domain>"

  Source B — Data_Dyamond_PostProcessed/<out_*>/ subdirectories:
    one STAC collection per out_* folder
    e.g. out_1_1 → collection "dyamond-out-1-1"
         out_9   → collection "dyamond-out-9"

Usage:
  python generate_catalog.py              # full index
  python generate_catalog.py --limit 5   # test mode (5 files per collection)
"""
import re, sys, pystac, xarray as xr
from pathlib import Path
from datetime import datetime, timezone
from shapely.geometry import mapping, box

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/users/nfarabul/data-compression/netCDF_files")
DYAMOND   = DATA_DIR / "Data_Dyamond_PostProcessed"   # symlink → /capstor/...
OUT       = Path("./stac_out")
BASE      = "http://localhost:8000"

LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

# ── variable metadata ──────────────────────────────────────────────────────────
# extend this dict whenever a new variable is discovered
VAR_META = {
    "geopot":    {"standard_name": "geopotential",                      "units": "m2 s-2"},
    "qv":        {"standard_name": "specific_humidity",                 "units": "kg kg-1"},
    "rh":        {"standard_name": "relative_humidity",                 "units": "%"},
    "t":         {"standard_name": "air_temperature",                   "units": "K"},
    "q":         {"standard_name": "specific_humidity",                 "units": "kg kg-1"},
    "t_q":       {"standard_name": "air_temperature+specific_humidity", "units": "mixed"},
    "t_s":       {"standard_name": "surface_temperature",               "units": "K"},
    "freshsnow": {"standard_name": "fresh_snow_density",                "units": "kg m-3"},
    "smi":       {"standard_name": "soil_moisture_index",               "units": "1"},
    "w_i":       {"standard_name": "cloud_ice_water_content",           "units": "kg kg-1"},
    "u":         {"standard_name": "eastward_wind",                     "units": "m s-1"},
    "v":         {"standard_name": "northward_wind",                    "units": "m s-1"},
    "w":         {"standard_name": "upward_air_velocity",               "units": "m s-1"},
    "temp":      {"standard_name": "air_temperature",                   "units": "K"},
    "pres":      {"standard_name": "air_pressure",                      "units": "Pa"},
    "tqv":       {"standard_name": "atmosphere_water_vapor_content",    "units": "kg m-2"},
    "cape":      {"standard_name": "convective_available_potential_energy", "units": "J kg-1"},
    "prec":      {"standard_name": "precipitation_flux",                "units": "kg m-2 s-1"},
    "olr":       {"standard_name": "toa_outgoing_longwave_flux",        "units": "W m-2"},
}

# ── filename parsers ───────────────────────────────────────────────────────────
# handles: remap_<var>_<DATE>.nc  and  remap_<var>_<LEVEL>hPa_<DATE>.nc
# var can contain underscores (e.g. t_s, w_i, t_q)
RE_REMAP = re.compile(
    r"^remap_"
    r"(?P<var>[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*?)"   # non-greedy var (allows t_s, w_i)
    r"(?:_(?P<level>\d+)hPa)?"
    r"_(?P<date>\d{8}T\d{6}Z)\.nc$"
)

RE_TIGGE = re.compile(
    r"^tigge_(?P<domain>[a-z]+)_(?P<vars>[a-z_q]+)"
    r"_dx=(?P<res>[\d.]+)"
    r"_(?P<yyyy>\d{4})_(?P<mm>\d{2})_(?P<dd>\d{2})\.nc$"
)

def parse_remap(path, rel_prefix="", out_name=""):
    m = RE_REMAP.match(path.name)
    if not m:
        return None
    dt  = datetime.strptime(m.group("date"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    var = m.group("var")
    if out_name:
        cid = f"dyamond-{out_name.replace('_', '-')}"
    else:
        cid = f"dyamond-direct-{var}"
    return {
        "convention":  "remap",
        "collection":  cid,
        "variable":    var,
        "level":       m.group("level"),
        "datetime":    dt,
        "resolution":  None,
        "domain":      None,
        "out_folder":  out_name or "direct",
        "rel_path":    f"{rel_prefix}{path.name}",
    }

def parse_tigge(path):
    m = RE_TIGGE.match(path.name)
    if not m:
        return None
    dt = datetime(int(m.group("yyyy")), int(m.group("mm")), int(m.group("dd")),
                  tzinfo=timezone.utc)
    return {
        "convention":  "tigge",
        "collection":  f"tigge-{m.group('domain')}",
        "variable":    m.group("vars"),
        "level":       None,
        "datetime":    dt,
        "resolution":  m.group("res"),
        "domain":      m.group("domain"),
        "out_folder":  None,
        "rel_path":    path.name,
    }

def parse_file(path, rel_prefix="", out_name=""):
    return parse_remap(path, rel_prefix, out_name) or parse_tigge(path)

def get_bbox(path):
    try:
        ds = xr.open_dataset(path, chunks={})
        lon = next((ds[v] for v in ds.coords if v in ("lon","longitude","rlon")), None)
        lat = next((ds[v] for v in ds.coords if v in ("lat","latitude","rlat")), None)
        if lon is not None and lat is not None:
            bbox = [float(lon.min()), float(lat.min()),
                    float(lon.max()), float(lat.max())]
            ds.close()
            return bbox
        ds.close()
    except Exception as e:
        pass   # silent fallback — bbox reading is slow, failure is common
    return [-180.0, -90.0, 180.0, 90.0]

def make_item(path, meta):
    bbox     = get_bbox(path)
    var_info = VAR_META.get(meta["variable"],
                            {"standard_name": meta["variable"], "units": "unknown"})
    props = {k: v for k, v in {
        "collection":     meta["collection"],
        "variable":       meta["variable"],
        "standard_name":  var_info["standard_name"],
        "units":          var_info["units"],
        "level":          meta["level"],
        "resolution_deg": meta["resolution"],
        "domain":         meta["domain"],
        "convention":     meta["convention"],
        "out_folder":     meta["out_folder"],
        "source_file":    path.name,
    }.items() if v is not None}

    item_id = path.stem.replace("=", "-").replace(" ", "_")
    item = pystac.Item(
        id=item_id,
        geometry=mapping(box(*bbox)),
        bbox=bbox,
        datetime=meta["datetime"],
        properties=props,
    )
    item.add_asset("data", pystac.Asset(
        href=f"{BASE}/files/{meta['rel_path']}",
        media_type="application/x-netcdf",
        roles=["data"],
        title=path.name,
        extra_fields={"file:local_path": str(path)},
    ))
    return item

# ── discover files ─────────────────────────────────────────────────────────────
collections_map: dict = {}   # collection_id → [(path, meta), ...]

SKIP_NAMES    = {"error", "Untitled.ipynb"}
SKIP_SUFFIXES = {".ipynb", ".nc_main"}

# --- Source A: direct .nc files in DATA_DIR -----------------------------------
print(f"\n[1/2] Scanning direct files in {DATA_DIR} ...")
direct = sorted(
    f for f in DATA_DIR.iterdir()
    if f.is_file()
    and f.suffix == ".nc"
    and f.name not in SKIP_NAMES
    and not any(f.name.endswith(s) for s in SKIP_SUFFIXES)
)
print(f"      {len(direct)} .nc file(s) found")

for path in direct:
    meta = parse_file(path)
    if meta is None:
        print(f"  [skip] unrecognised: {path.name}")
        continue
    collections_map.setdefault(meta["collection"], []).append((path, meta))

# --- Source B: out_*/ directories directly inside DATA_DIR ------------------
out_dirs = sorted(
    d for d in DATA_DIR.iterdir()
    if d.is_dir() and d.name.startswith("out_")
)
print(f"\n[2/2] Scanning {len(out_dirs)} out_* folders in {DATA_DIR} ...")

if not out_dirs:
    print("  [warn] no out_* folders found in DATA_DIR")
else:
    for out_dir in out_dirs:
        out_name = out_dir.name   # e.g. "out_1_1", "out_9"
        rel_prefix = f"{out_name}/"

        nc_files = sorted(
            f for f in out_dir.iterdir()
            if f.is_file()
            and f.suffix == ".nc"
            and f.name not in SKIP_NAMES
            and not f.name.endswith(".nc_main")
        )
        if not nc_files:
            print(f"  {out_name}: 0 files, skipping")
            continue

        # discover all variables in this folder
        vars_found = set()
        for f in nc_files:
            m = RE_REMAP.match(f.name)
            if m:
                vars_found.add(m.group("var"))

        print(f"  {out_name}: {len(nc_files)} files  vars={sorted(vars_found)}")

        for path in nc_files:
            meta = parse_file(path, rel_prefix=rel_prefix, out_name=out_name)
            if meta is None:
                continue
            collections_map.setdefault(meta["collection"], []).append((path, meta))

# ── build catalog ──────────────────────────────────────────────────────────────
total = sum(len(v) for v in collections_map.values())
print(f"\nTotal: {len(collections_map)} collections, {total} files")
if LIMIT:
    print(f"       (limited to {LIMIT} items per collection for testing)")

catalog = pystac.Catalog(
    id="dyamond-catalog",
    description="DYAMOND post-processed outputs — all variables, all runs",
    title="DYAMOND Catalog",
)

for cid, entries in sorted(collections_map.items()):
    if LIMIT:
        entries = entries[:LIMIT]

    dates     = [m["datetime"] for _, m in entries]
    variables = sorted({m["variable"] for _, m in entries})

    collection = pystac.Collection(
        id=cid,
        description=f"{cid}  |  variables: {', '.join(variables)}  |  {len(entries)} items",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
            temporal=pystac.TemporalExtent([[min(dates), max(dates)]]),
        ),
        extra_fields={
            "variables":  variables,
            "out_folder": entries[0][1]["out_folder"],
        },
    )

    print(f"  '{cid}': {len(entries)} items  vars={variables}", end="", flush=True)
    skipped = 0
    for i, (path, meta) in enumerate(entries):
        try:
            item = make_item(path, meta)
            collection.add_item(item)
        except Exception as e:
            skipped += 1
        if (i+1) % 500 == 0:
            print(f" {i+1}...", end="", flush=True)
    if skipped:
        print(f" ({skipped} skipped)", end="")
    print(" ✓")

    catalog.add_child(collection)

# ── write output ───────────────────────────────────────────────────────────────
OUT.mkdir(parents=True, exist_ok=True)
catalog.normalize_hrefs(str(OUT))
catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

print(f"\n✓  Catalog written to {OUT}/")
print(f"   Collections : {len(list(catalog.get_children()))}")
print(f"   Total items : {len(list(catalog.get_all_items()))}")