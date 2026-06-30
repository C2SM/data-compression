# app.py  —  STAC API serving Data_Dyamond_PostProcessed out_* collections
import json
import urllib.parse
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

STAC_OUT = Path("/users/nfarabul/data-compression/stac_out")
DATA_DIR = Path("/users/nfarabul/data-compression/netCDF_files")
BASE     = "http://localhost:8000"

app = FastAPI(title="DYAMOND STAC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)

# ── file download ──────────────────────────────────────────────────────────────
# Handles paths like:
#   /files/remap_qv_20200224T000000Z.nc
#   /files/Data_Dyamond_PostProcessed/out_1_1/remap_geopot_20200120T000000Z.nc
#   /files/Data_Dyamond_PostProcessed/out_9/remap_t_s_20220225T000000Z.nc
@app.get("/files/{file_path:path}")
def download_file(file_path: str):
    decoded = urllib.parse.unquote(file_path)

    # block path traversal
    if ".." in decoded:
        raise HTTPException(403, "Access denied")

    path = DATA_DIR / decoded

    if not path.exists():
        raise HTTPException(404, f"File not found: {decoded}")
    if not path.is_file():
        raise HTTPException(400, f"Not a file: {decoded}")

    return FileResponse(
        path=str(path.resolve()),
        media_type="application/x-netcdf",
        filename=path.name,
    )

# ── conformance ───────────────────────────────────────────────────────────────
@app.get("/conformance")
def conformance():
    return {
        "conformsTo": [
            "https://api.stacspec.org/v1.0.0/core",
            "https://api.stacspec.org/v1.0.0/item-search",
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
        ]
    }

# ── root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    data = load_json(STAC_OUT / "catalog.json")
    data["stac_api_version"] = "1.0.0"
    data["type"] = "Catalog"
    # conformsTo in root body — required by STAC Browser v5
    data["conformsTo"] = [
        "https://api.stacspec.org/v1.0.0/core",
        "https://api.stacspec.org/v1.0.0/item-search",
        "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
        "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas30",
    ]
    data["links"] = [
        {"rel": "self",        "href": f"{BASE}/",            "type": "application/json"},
        {"rel": "root",        "href": f"{BASE}/",            "type": "application/json"},
        {"rel": "conformance", "href": f"{BASE}/conformance", "type": "application/json"},
        {"rel": "data",        "href": f"{BASE}/collections", "type": "application/json"},
        {"rel": "items",       "href": f"{BASE}/search",      "type": "application/geo+json"},
        {"rel": "search",      "href": f"{BASE}/search",      "type": "application/json", "method": "GET"},
        {"rel": "search",      "href": f"{BASE}/search",      "type": "application/json", "method": "POST"},
    ]
    return data

# ── collections ───────────────────────────────────────────────────────────────
@app.get("/collections")
def get_collections():
    cols = []
    for p in sorted(STAC_OUT.iterdir()):
        if p.is_dir() and (p / "collection.json").exists():
            col = load_json(p / "collection.json")
            col["links"] = _col_links(col["id"])
            cols.append(col)
    return {
        "collections":    cols,
        "numberMatched":  len(cols),
        "numberReturned": len(cols),
        "links": [
            {"rel": "self", "href": f"{BASE}/collections", "type": "application/json"},
            {"rel": "root", "href": f"{BASE}/",            "type": "application/json"},
        ],
    }

@app.get("/collections/{collection_id}")
def get_collection(collection_id: str):
    path = STAC_OUT / collection_id / "collection.json"
    if not path.exists():
        raise HTTPException(404, f"Collection '{collection_id}' not found")
    col = load_json(path)
    col["links"] = _col_links(collection_id)
    return col

def _col_links(cid):
    return [
        {"rel": "self",   "href": f"{BASE}/collections/{cid}",       "type": "application/json"},
        {"rel": "root",   "href": f"{BASE}/",                        "type": "application/json"},
        {"rel": "parent", "href": f"{BASE}/",                        "type": "application/json"},
        {"rel": "items",  "href": f"{BASE}/collections/{cid}/items", "type": "application/geo+json"},
    ]

# ── items ─────────────────────────────────────────────────────────────────────
@app.get("/collections/{collection_id}/items")
def get_items(
    collection_id: str,
    limit:    int = Query(100, ge=1, le=10000),
    offset:   int = Query(0,   ge=0),
    variable: str = Query(None),
    datetime: str = Query(None),
):
    col_dir = STAC_OUT / collection_id
    if not col_dir.exists():
        raise HTTPException(404, f"Collection '{collection_id}' not found")

    all_items = []
    for p in sorted(col_dir.rglob("*.json")):
        if p.name == "collection.json" or ".ipynb_checkpoints" in str(p):
            continue
        item = load_json(p)
        # apply optional filters
        if variable and item.get("properties", {}).get("variable") != variable:
            continue
        if datetime:
            item_dt = item.get("properties", {}).get("datetime", "")
            if "/" in datetime:
                dt_from, dt_to = datetime.split("/")
                if not (dt_from <= item_dt <= dt_to):
                    continue
            else:
                if item_dt[:10] != datetime[:10]:
                    continue
        item["links"] = _item_links(collection_id, item["id"])
        all_items.append(item)

    page = all_items[offset: offset + limit]
    links = [
        {"rel": "self",       "href": f"{BASE}/collections/{collection_id}/items?limit={limit}&offset={offset}", "type": "application/geo+json"},
        {"rel": "root",       "href": f"{BASE}/",                                                                "type": "application/json"},
        {"rel": "collection", "href": f"{BASE}/collections/{collection_id}",                                     "type": "application/json"},
    ]
    if offset + limit < len(all_items):
        links.append({"rel": "next",
                      "href": f"{BASE}/collections/{collection_id}/items?limit={limit}&offset={offset+limit}",
                      "type": "application/geo+json"})
    if offset > 0:
        links.append({"rel": "prev",
                      "href": f"{BASE}/collections/{collection_id}/items?limit={limit}&offset={max(0,offset-limit)}",
                      "type": "application/geo+json"})
    return {
        "type":           "FeatureCollection",
        "features":       page,
        "numberMatched":  len(all_items),
        "numberReturned": len(page),
        "links":          links,
    }

@app.get("/collections/{collection_id}/items/{item_id}")
def get_item(collection_id: str, item_id: str):
    path = STAC_OUT / collection_id / item_id / f"{item_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Item '{item_id}' not found")
    item = load_json(path)
    item["links"] = _item_links(collection_id, item_id)
    return item

def _item_links(cid, item_id):
    return [
        {"rel": "self",       "href": f"{BASE}/collections/{cid}/items/{item_id}", "type": "application/geo+json"},
        {"rel": "root",       "href": f"{BASE}/",                                  "type": "application/json"},
        {"rel": "parent",     "href": f"{BASE}/collections/{cid}/items",           "type": "application/geo+json"},
        {"rel": "collection", "href": f"{BASE}/collections/{cid}",                 "type": "application/json"},
    ]

# ── search ────────────────────────────────────────────────────────────────────
def _load_all_items():
    items = []
    for p in sorted(STAC_OUT.rglob("*.json")):
        if p.name in ("catalog.json", "collection.json"):
            continue
        if ".ipynb_checkpoints" in str(p):
            continue
        item = load_json(p)
        if item.get("type") == "Feature":
            items.append(item)
    return items

def _filter(items, collections=None, variable=None, datetime=None,
            convention=None, out_folder=None, bbox=None):
    results = []
    for item in items:
        props = item.get("properties", {})

        if collections:
            col_ids = [c.strip() for c in collections.split(",")]
            if item.get("collection") not in col_ids:
                continue
        if variable   and props.get("variable")   != variable:   continue
        if convention and props.get("convention")  != convention: continue
        if out_folder and props.get("out_folder")  != out_folder: continue
        if datetime:
            item_dt = props.get("datetime", "")
            if "/" in datetime:
                dt_from, dt_to = datetime.split("/")
                if not (dt_from <= item_dt <= dt_to):
                    continue
            else:
                if item_dt[:10] != datetime[:10]:
                    continue
        if bbox and len(bbox) == 4:
            ib = item.get("bbox", [])
            if len(ib) == 4:
                if ib[2] < bbox[0] or ib[0] > bbox[2] or ib[3] < bbox[1] or ib[1] > bbox[3]:
                    continue
        results.append(item)
    return results

@app.get("/search")
def search_get(
    collections: str = Query(None),
    variable:    str = Query(None),
    datetime:    str = Query(None),
    convention:  str = Query(None),
    out_folder:  str = Query(None),
    bbox:        str = Query(None),
    limit:       int = Query(100, ge=1, le=10000),
):
    bbox_list = [float(x) for x in bbox.split(",")] if bbox else None
    results = _filter(_load_all_items(), collections, variable,
                      datetime, convention, out_folder, bbox_list)
    return _search_response(results[:limit], len(results))

@app.post("/search")
async def search_post(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    cols       = ",".join(body.get("collections", [])) or None
    var        = body.get("filter", {}).get("args", [{}]*2)[1] if body.get("filter") else None
    dt         = body.get("datetime")
    bbox       = body.get("bbox")
    out_folder = body.get("out_folder")
    limit      = body.get("limit", 100)
    results = _filter(_load_all_items(), cols, var, dt, out_folder=out_folder, bbox=bbox)
    return _search_response(results[:limit], len(results))

def _search_response(features, total):
    return {
        "type":           "FeatureCollection",
        "features":       features,
        "numberMatched":  total,
        "numberReturned": len(features),
        "links": [{"rel": "self", "href": f"{BASE}/search", "type": "application/geo+json"}],
    }