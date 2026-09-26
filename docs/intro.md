# dc_toolkit from zero to hero

This guide takes you from an empty laptop to compressed climate data you can trust. You will install
`dc_toolkit`, let it search for the best way to compress a field (`evaluate_combos`), write the compressed
store (`compress`), check what the compression did to your data, and then learn the shortcuts: compressing
with a pipeline of your own, using EBCC, and keeping physics safe.

Everything runs on the small sample file that ships with the repository, so you can follow along without
any data of your own. Each step takes seconds.

**Contents**

1. [Install](#1-install)
2. [The ideas in five minutes](#2-the-ideas-in-five-minutes)
3. [Your first sweep: `evaluate_combos`](#3-your-first-sweep-evaluate_combos)
4. [Your first store: `compress`](#4-your-first-store-compress)
5. [Check the real error, then choose your budget](#5-check-the-real-error-then-choose-your-budget)
6. [The full sweep and the whole file](#6-the-full-sweep-and-the-whole-file)
7. [Compress without a sweep: a pipeline of your own](#7-compress-without-a-sweep-a-pipeline-of-your-own)
8. [EBCC](#8-ebcc)
9. [Steering the search](#9-steering-the-search)
10. [Your own files on a laptop](#10-your-own-files-on-a-laptop)
11. [Troubleshooting](#11-troubleshooting)
12. [Cheat sheet](#12-cheat-sheet)

---

## 1. Install

### 1.1 What you need first

| You need | Why | macOS (Homebrew), for example | Ubuntu / Debian, for example |
|---|---|---|---|
| Python 3.11 or newer | the toolkit | `brew install python` | `sudo apt install python3 python3-venv python3-dev` |
| A C compiler | builds `mpi4py` | `xcode-select --install` | `sudo apt install build-essential` |
| An MPI library with `mpicc` | `evaluate_combos` is an MPI program, even with one process | `brew install open-mpi` | `sudo apt install libopenmpi-dev openmpi-bin` |
| ecCodes (only for GRIB input) | reading `.grib` files | `brew install eccodes` | `sudo apt install libeccodes-dev` |

On Windows, use WSL2 with Ubuntu and follow the Ubuntu column.

You do **not** need a cluster or `srun`. `evaluate_combos` is an MPI program: on a laptop start it with
`mpirun -n <cores>`, one process per physical core (Open MPI refuses more); the processes share one copy of
the sample. Every other command is started directly and runs as a single process.

### 1.2 Install the toolkit

```bash
git clone https://github.com/C2SM/data-compression.git dc_toolkit   # or the SSH URL if you use SSH keys
cd dc_toolkit
python3 -m venv venv
source venv/bin/activate
bash install_dc_toolkit.sh
```

The script installs the package in editable mode and builds `mpi4py` against your MPI. If it stops while
looking for `mpicc`, the MPI library from the table is missing.

### 1.3 Check that it works

```bash
dc_toolkit --help
```

You should see the list of commands, among them `evaluate_combos` and `compress`. `dc_toolkit COMMAND --help`
documents every option of a command, and `dc_toolkit help` prints all of them at once.

### 1.4 In every new terminal: activate and pin the threads

```bash
source venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1 \
       NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1
```

The sweep runs one process per core, each evaluating one pipeline at a time. If the compression libraries
also started their own threads, they would fight over the same cores, so `evaluate_combos` and `compress`
refuse to start until they are pinned to 1:

```text
[oversubscription-check] WARNING: codec-internal thread variables not pinned to 1:
  - OMP_NUM_THREADS=<unset>
  ...
  Suggested: export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ...
  Aborting (use --no-oversubscription-check to override).
```

If you see this, run the `export` line above and try again. Putting that line in your shell profile saves
you from meeting the message twice.

For the rest of the guide, two shortcuts keep the commands short:

```bash
FILE="netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc"   # the bundled sample: temperature t and humidity q
OUT="$HOME/dc_tutorial"                               # where the tutorial writes
```

---

## 2. The ideas in five minutes

### A pipeline has three stages

`dc_toolkit` stores data as [Zarr](https://zarr.dev) arrays. Each array is written through a **pipeline** of
up to three codecs, applied in this order:

| Stage | Job | Lossy? | Examples |
|---|---|---|---|
| **filter** | reshape the *values* so they compress better | usually yes | `bitround` (keep N mantissa bits), `quantize` (keep N decimal digits), `delta`, `fixedscaleoffset`, `astype` |
| **serializer** | turn the array into bytes | depends | plain bytes and `pcodec` are lossless; `zfpy` and `EBCC` are lossy |
| **compressor** | squeeze the bytes | never | `zstd`, `blosc`, `lzma`, `bz2`, `zlib`, `lz4` |

Any stage can be absent. A pipeline is written down as a small JSON object whose entries are the codecs'
own Zarr JSON (the form the array's `zarr.json` records them in):

```json
{
  "compressor": {"name": "numcodecs.zstd",     "configuration": {"level": 9}},
  "filter":     {"name": "numcodecs.bitround", "configuration": {"keepbits": 12}},
  "serializer": null
}
```

In logs and tables the same pipeline is printed as `zstd(level=9) | bitround(keepbits=12) | -`, always in the
order **compressor | filter | serializer**, with `-` for an absent stage.

### The two commands

```text
   your file.nc
        │
        ▼
  evaluate_combos      tries thousands of pipelines on a sample of each field,
        │              measures ratio and error, keeps those within your error budget
        ▼
  results_{var}.parquet   every pipeline that was tried
  manifest_{var}.json     the winner
        │
        ▼
     compress          writes the whole field with the winner, reads it back,
        │              and re-checks the error on the real store
        ▼
  your file.zarr
```

`evaluate_combos` never writes your data; it only measures. `compress` never searches; it writes what the
sweep found, or what you tell it to (section 7).

### The error budget

Lossy compression changes your values, so you decide by how much. The one mandatory setting is
**`--l1-threshold`**: the allowed *relative mean absolute error*. `0.005` means "on average, values may be off
by 0.5 % of their typical magnitude". The other gates are derived from it unless you set them yourself:

| Gate | What it limits | Default |
|---|---|---|
| L1 | mean absolute error | your `--l1-threshold` |
| L2 | root-mean-square error | 2 × L1 |
| Linf | the **worst single cell** | 10 × L1 |
| bias | systematic drift (mean signed error) | 0.5 × L1 |

One more gate needs no setting: a pipeline that turns a missing value (`NaN`) of the sample into a number, or
a number into `NaN` or `Inf`, is dropped (the `n_corrupt` count).

A pipeline is **kept** only if it passes every gate, and the winner is the kept pipeline with the highest
compression ratio. All errors are relative to the magnitude of the field, which matters more than you might
expect; section 5 shows why.

---

## 3. Your first sweep: `evaluate_combos`

```bash
mpirun -n 8 dc_toolkit evaluate_combos "$FILE" \
    --where-to-write "$OUT/sweep" \
    --field-to-compress t \
    --l1-threshold 0.005 \
    --max-evals 300
```

| Part | Meaning |
|---|---|
| `mpirun -n 8` | one process per core, here eight; use your machine's number of physical cores |
| `"$FILE"` | the file to read: `.nc`, `.grib` or `.zarr`, recognised by its extension |
| `--where-to-write` | the directory for the results; created if missing |
| `--field-to-compress t` | sweep only the variable `t`; leave it out to sweep every field of the file |
| `--l1-threshold 0.005` | the error budget (required) |
| `--max-evals 300` | try 300 pipelines spread evenly over the grid (the same 300 on every run): a quick first look |

The interesting lines of the output:

```text
[combo-filter] skipped 7920 unsupported filter/serializer pairing(s) (FixedScaleOffset->ZFPY, BitRound->ZFPY below the mantissa width).
[max-evals] evaluating a uniform subset of 300 of the 17127 combos.
[topology] 1 node(s) x 8 rank(s)/node = 8 parallel evaluations, one shared sample per node.
[memory] ... = 1.0 GiB total.
[sweep] 300 combos: 300 from the 33 x 23 x 33 grid (valid pairings, --max-evals 300) + 0 EBCC; 300 to evaluate, ~300 per node, claimed by its ranks as they free up.
[sweep] consolidated the per-rank CSVs -> .../sweep/results_t.parquet (300 row(s)).
best pipeline: lzma(preset=9) | bitround(keepbits=52) | zfpy_flat(mode=2, rate=8)
Compression Ratio: 37.355 | Relative L1 Error: 3.357e-03 | Euclidean Distance: 1.803e+02
```

Reading it top to bottom: the full search space for this field is 33 compressors × 23 filters × 33
serializers; pairings that cannot work are removed up front; your laptop evaluates one pipeline per core
(a process that finishes one takes the next from a counter the processes share); and the best pipeline
within the budget shrinks `t` by a factor of 37.

On a laptop the output usually also has a `[memcheck] auto-shrunk sample budget ...` line above these: the
default budget of 5 GB does not fit beside eight working sets in a laptop's memory, so the sweep lowers it.
That is expected; section 10 explains the memory.

### What was written

| File in `$OUT/sweep` | What it is |
|---|---|
| `results_t.parquet` | **every** pipeline tried: ratio, all error metrics, a verdict per gate, and `keep` |
| `manifest_t.json` | the winner, the thresholds that were applied, and the sweep's settings; `compress` reads this |
| `config_space_t.csv` | the list of planned pipelines, in the order the processes claim them |
| `config_space_t_rank0.csv` … `config_space_t_rank7.csv` | one per process: the results as they were produced; this is what lets an interrupted sweep resume |
| `failures_t_rank0.csv` … `failures_t_rank7.csv` | one per process: the pipelines that raised an error, with the error (just a header when none failed) |
| `sweep_state_t.json` | what the results were measured on; if it changes, the sweep starts over instead of resuming |

### Look at the results yourself

The winner:

```bash
python -c "
import json
m = json.load(open('$OUT/sweep/manifest_t.json'))
print(m['best']['name'])
print('ratio', round(m['best']['ratio'], 2), '| kept', m['num_passed'], 'of', m['num_combos'])
"
```

Everything that was tried, best first:

```python
import os, pandas as pd

df = pd.read_parquet(os.path.expanduser("~/dc_tutorial/sweep/results_t.parquet"))
kept = df[df["keep"]].sort_values("ratio", ascending=False)
print(len(df), "tried,", len(kept), "kept")
print(kept[["name", "ratio", "l1_rel", "linf_rel"]].head())
```

```text
300 tried, 280 kept
                                                              name      ratio    l1_rel  linf_rel
lzma(preset=9) | bitround(keepbits=52) | zfpy_flat(mode=2, rate=8)  37.354618  0.003357  0.044996
...
```

Keep an eye on that last column: `linf_rel` says the worst cell is off by 4.5 %. Section 5 comes back to it.

The columns worth knowing: `ratio`, the relative errors `l1_rel`, `l2_rel`, `linf_rel`, `bias_rel`, one
`pass_*` column per gate, and `keep`. The `pipeline` column holds the JSON of each row, ready to be reused
(section 7).

---

## 4. Your first store: `compress`

```bash
dc_toolkit compress "$FILE" "$OUT/sweep"
```

`compress` takes the **same input file** and the **directory of the sweep**. It finds `manifest_t.json` there,
writes `t` with the winning pipeline, and checks its own work:

```text
[compress] (1/1) t from manifest_t.json: lzma(preset=9) | bitround(keepbits=52) | zfpy_flat(mode=2, rate=8)
[chunks] t: inner_chunk_mib=16 (manifest), max_inner_chunk_mib=256 (manifest), spatial_split=True (manifest)
[compress] t: ... -> ratio=37.355 L1_rel=3.357e-03 eucd=1.803e+02  (0.2s)
[verify-gate] t: PASS, production error norms are within the sweep thresholds.
[cr-drift] t: PASS (achieved 37.35x vs predicted 37.35x, drift +0.0%)
[compress] consolidated metadata on .../sweep/tigge_pl_t_q_dx=2_2024_08_02.zarr (1 array(s): t)
```

Three safety nets are at work here:

- **The verify gate.** `compress` reads back every part of the array right after writing it and computes the
  errors of the whole field from the real store. If they exceed the sweep's thresholds, or a missing value
  came back as a number, the field fails.
- **CR drift.** The sweep measured a *sample*; this line compares the ratio it predicted with the one the
  whole field achieved. A large drift means the sample did not represent the field well.
- **Staging.** A field is written into a staging store beside the real one and moved in only after its
  checks passed. An interrupted or failed write never leaves a half-written array behind, and never replaces
  a good one.

The store is `$OUT/sweep/<your file's name without its extension>.zarr`, with one array per field. `batch_manifest.json` records
what happened to each field. If any field fails, `compress` exits with status 1, so scripts notice.

### Open the store

```python
import os, xarray as xr

store = os.path.expanduser("~/dc_tutorial/sweep/tigge_pl_t_q_dx=2_2024_08_02.zarr")
print(xr.open_zarr(store)["t"])
```

`dc_toolkit open_zarr_and_inspect "$OUT/sweep/tigge_pl_t_q_dx=2_2024_08_02.zarr"` prints the codecs, the
chunking and the achieved ratio of every array, plus the first few values.

---

## 5. Check the real error, then choose your budget

Do not skip this section. Compare the store with the original:

```python
import os, numpy as np, xarray as xr

out = os.path.expanduser("~/dc_tutorial")
orig = xr.open_dataset("netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc")["t"].values
comp = xr.open_zarr(f"{out}/sweep/tigge_pl_t_q_dx=2_2024_08_02.zarr")["t"].values
print("largest error:", float(np.abs(orig - comp).max()), "K")
```

```text
largest error: 14.37 K
```

A ratio of 37 sounded great; a temperature that is wrong by 14 K does not. Nothing malfunctioned. The
budget was `0.005`, *relative to the magnitude of the field*. Temperature is stored in kelvin, around 280, so
0.5 % is 1.4 K on average, and the worst-cell gate (10 × L1 = 5 % of the field's largest value, about
320 K) allows 16 K. The budget was simply far too loose for this field.

**A budget that suits one field can ruin another.** Think about what an error means in the units of *your*
field, and always look at the decoded data once before you trust a setting.

So tighten the budget and run the same command again:

```bash
mpirun -n 8 dc_toolkit evaluate_combos "$FILE" --where-to-write "$OUT/sweep" --field-to-compress t \
    --l1-threshold 0.0001 --max-evals 300
```

```text
[resume] 300 of 300 combo(s) of 't' are already recorded; skipping those.
best pipeline: blosc(clevel=9, cname=zstd, shuffle=0, typesize=8) | bitround(keepbits=52) | zfpy(mode=4, tolerance=0.25)
Compression Ratio: 7.635 | Relative L1 Error: 6.034e-05 | Euclidean Distance: 2.760e+00
```

Note the first line: nothing was measured again. The 300 results were already on disk, so the sweep only
re-applied the gates with the new budget and picked a new winner. Trying different budgets is therefore
cheap. The new winner uses zfp's fixed-accuracy mode with a tolerance of 0.25, which bounds the error of
every cell by 0.25 K, at a ratio of 7.6.

Write it:

```bash
dc_toolkit compress "$FILE" "$OUT/sweep"
```

The store already holds a `t`, written with the old winner; `compress` sees that the pipeline changed and
rewrites it (`[compress] t in ...: its pipeline, verify gate changed; rewriting it.`). A field is skipped only when the store
holds it exactly as the run would write it.

Run the comparison again: the largest error is now 0.09 K.

---

## 6. The full sweep and the whole file

`--max-evals` was only there for a fast first look. Drop it to search the entire space:

```bash
mpirun -n 8 dc_toolkit evaluate_combos "$FILE" --where-to-write "$OUT/full" --field-to-compress t --l1-threshold 0.0001
```

```text
[sweep] 17127 combos: 17127 from the 33 x 23 x 33 grid (valid pairings) + 0 EBCC; 17127 to evaluate, ~17127 per node, claimed by its ranks as they free up.
best pipeline: bz2(level=3) | quantize(digits=1, dtype=float64) | -
Compression Ratio: 7.866 | Relative L1 Error: 5.486e-05 | Euclidean Distance: 2.309e+00
```

All 17 127 pipelines for `t`, in under a minute on eight cores, because the sample field is tiny; on real
fields expect minutes to hours, depending on the sample size (section 10). The full search also found a
better pipeline than the quick look did: 7.87 against 7.64 at the same budget. `--max-evals` is for getting
your bearings, not for the final answer.

Drop `--field-to-compress` as well and every field of the file is swept, one after the other, each with
its own results and manifest; `compress` then writes them all into one store:

```bash
mpirun -n 8 dc_toolkit evaluate_combos "$FILE" --where-to-write "$OUT/both" --l1-threshold 0.0005 --max-evals 300
dc_toolkit compress "$FILE" "$OUT/both"
```

```text
[compress] will compress 2 field(s): q, t
[verify-gate] q: PASS, production error norms are within the sweep thresholds.
[verify-gate] t: PASS, production error norms are within the sweep thresholds.
[compress] consolidated metadata on .../both/tigge_pl_t_q_dx=2_2024_08_02.zarr (2 array(s): q, t)
```

One budget for all fields is convenient but rarely right (section 5). For real work, run one sweep per
field into the same directory, each with the budget that suits it, and compress once at the end. To write
only some fields, use `compress ... --vars t,q`.

Helper variables that describe the grid rather than the weather (CF "bounds" such as `lat_bnds` or
`clon_bnds`) are left out of a whole-file sweep, because lossy compression would move the grid. The log says
so when it happens.

---

## 7. Compress without a sweep: a pipeline of your own

You do not have to run `evaluate_combos`. If you already know how you want a field stored, give `compress`
the pipeline directly with **`--pipeline`**, and name the field(s) with **`--vars`**.

### From a file

```bash
mkdir -p "$OUT"
cat > "$OUT/lossless.json" <<'EOF'
{
  "compressor": {"name": "numcodecs.zstd",   "configuration": {"level": 9}},
  "filter":     null,
  "serializer": {"name": "numcodecs.pcodec", "configuration": {"level": 8}}
}
EOF

dc_toolkit compress "$FILE" "$OUT/manual_lossless" --vars t --pipeline "$OUT/lossless.json"
```

```text
[compress] t: zstd(level=9) | - | pcodec(level=8) -> ratio=2.182 L1_rel=0.000e+00 eucd=0.000e+00  (0.2s)
```

No filter and a lossless serializer: the error is exactly zero, and the ratio is 2.2.

### A lossy pipeline, with a budget to check it against

There is no sweep here, hence no thresholds for the verify gate. Supply the budget yourself:

```bash
cat > "$OUT/lossy.json" <<'EOF'
{
  "compressor": {"name": "numcodecs.zstd",     "configuration": {"level": 9}},
  "filter":     {"name": "numcodecs.bitround", "configuration": {"keepbits": 12}},
  "serializer": null
}
EOF

dc_toolkit compress "$FILE" "$OUT/manual_lossy" --vars t --pipeline "$OUT/lossy.json" --l1-threshold 0.0005
```

```text
[compress] t: zstd(level=9) | bitround(keepbits=12) | - -> ratio=5.640 L1_rel=5.224e-05 eucd=2.225e+00  (0.2s)
[verify-gate] t: PASS, production error norms are within the sweep thresholds.
```

Without `--l1-threshold` the field is still written and its errors are still printed, but they are not
enforced (`the error norms are advisory`); a missing value that comes back as a number still fails the field.
With a budget the pipeline cannot meet, the field fails, nothing is left in the store, and the exit status
is 1:

```text
[verify-gate] FAIL: t: verify gate FAILED (pass_l1, pass_l2, pass_linf, pass_bias) | L1=5.224e-05 ... | thresholds: l1=1.000e-07 ...
```

As in a sweep, the L2, Linf and bias budgets default to 2 ×, 10 × and 0.5 × `--l1-threshold`;
`--l2-threshold`, `--linf-threshold` and `--bias-threshold` set them yourself.

### Inline, and several fields at once

```bash
dc_toolkit compress "$FILE" "$OUT/manual_inline" --vars t,q \
    --pipeline '{"compressor": {"name": "numcodecs.zstd", "configuration": {"level": 9}}, "filter": null, "serializer": {"name": "numcodecs.pcodec", "configuration": {"level": 8}}}'
```

### Reuse the winner of an earlier sweep

A manifest is accepted as it is, so the result of one sweep can be applied to another file, or to the next
time step, without searching again:

```bash
dc_toolkit compress "$FILE" "$OUT/reuse" --vars t --pipeline "$OUT/sweep/manifest_t.json" --l1-threshold 0.0001
```

Only the pipeline is taken from the manifest, not its thresholds, so pass the budget again (the worst-cell
budget follows from it, 10 × L1): the verify gate is what catches a pipeline that does not suit the new data.
A `fixedscaleoffset` filter, for one, is fitted to the value range of the swept field and does not clip;
`compress` refuses it when the new field's values do not fit that range.

### Use a row that is not the winner

Perhaps you want the best *lossless* pipeline, or the best one that avoids a certain codec. Every row of the
parquet carries its pipeline:

```python
import json, os, pandas as pd

out = os.path.expanduser("~/dc_tutorial")
df = pd.read_parquet(f"{out}/sweep/results_t.parquet")
row = df[df["keep"] & (df["l1_rel"] == 0)].sort_values("ratio", ascending=False).iloc[0]   # zero error on the sample
print(row["name"], row["ratio"])
json.dump(json.loads(row["pipeline"]), open(f"{out}/picked.json", "w"), indent=1)
```

Then `dc_toolkit compress "$FILE" "$OUT/picked" --vars t --pipeline "$OUT/picked.json"`.

### Codec snippets

The easiest source of correct JSON is a parquet row or a manifest. For writing by hand, these are the
common ones:

| Codec | JSON |
|---|---|
| zstd | `{"name": "numcodecs.zstd", "configuration": {"level": 9}}` (1 to 22) |
| lzma | `{"name": "numcodecs.lzma", "configuration": {"preset": 6}}` (0 to 9) |
| bz2 | `{"name": "numcodecs.bz2", "configuration": {"level": 9}}` (1 to 9) |
| bitround | `{"name": "numcodecs.bitround", "configuration": {"keepbits": 12}}` |
| quantize | `{"name": "numcodecs.quantize", "configuration": {"digits": 3, "dtype": "float64"}}` (the field's dtype) |
| pcodec | `{"name": "numcodecs.pcodec", "configuration": {"level": 8}}` |
| zfp, fixed accuracy | `{"name": "numcodecs.zfpy", "configuration": {"mode": 4, "tolerance": 0.25}}` (absolute error bound, in the field's units) |
| zfp, fixed rate | `{"name": "numcodecs.zfpy", "configuration": {"mode": 2, "rate": 8}}` (bits per value) |
| plain bytes / no stage | `null` |

Some pairings are refused because they cannot produce valid data: `bitround` in front of `zfpy` (unless
`keepbits` keeps the full mantissa, 23 for float32 and 52 for float64), `fixedscaleoffset` in front of `zfpy`
(or of `pcodec` when it packs into 8 bits), and anything but the `astype` cast around EBCC, which runs alone.
`compress` tells you when a pipeline is one of these.

---

## 8. EBCC

[EBCC](https://github.com/spcl/EBCC) is an error-bounded compressor built for climate fields: a JPEG 2000
image of each horizontal frame plus a correction that guarantees a **maximum absolute error**. At loose
error bounds it often beats everything else; it is also slow to encode, so it is optional and off by
default.

### Install it

EBCC is compiled on your machine. It needs `cmake` and the HDF5 headers (`brew install cmake hdf5`, or
`sudo apt install cmake libhdf5-dev`), then:

```bash
pip install -e ".[ebcc]"          # inside the venv; about ten minutes
```

### When it applies

- float fields whose last two dimensions are a regular **(lat, lon)** grid; unstructured grids such as ICON's
  native cells do not qualify;
- each side of the frame at least 32 points, and a side above 2047 needs a divisor between 32 and 2047 (the
  tile);
- no `NaN` or `Inf` anywhere in the field (the toolkit checks first, because the library would otherwise
  terminate the program: a sweep leaves EBCC out when the field holds any, and `compress` checks again);
- float64 fields are stored as float32, through an `astype` filter that a sweep adds for you; a pipeline of
  your own must carry it (see "EBCC without any sweep");
- it runs alone: no other filter, no compressor.

### Add EBCC to a sweep

```bash
mpirun -n 8 dc_toolkit evaluate_combos "$FILE" --where-to-write "$OUT/sweep_ebcc" --field-to-compress t \
    --l1-threshold 0.0005 --max-evals 300 --with-ebcc
```

```text
[ebcc] t: tile 91x180
[sweep] 307 combos: 300 from the 33 x 23 x 33 grid (valid pairings, --max-evals 300) + 7 EBCC; 307 to evaluate, ~307 per node, claimed by its ranks as they free up.
best pipeline: - | astype(decode_dtype=float64, encode_dtype=float32) | EBCC(height=91, width=180, base_cr=2, max_error_target=0.259055)
Compression Ratio: 12.034 | Relative L1 Error: 1.417e-04 | Euclidean Distance: 6.439e+00
```

`--with-ebcc` adds seven EBCC pipelines next to the regular ones, with error bounds from 10 % down to 0.01 %
of the field's value range; `--max-evals` never removes them. Here EBCC wins with a ratio of 12.0; the best
regular pipeline at the same budget reaches 9.3 in a full sweep.

### Sweep EBCC alone

```bash
mpirun -n 8 dc_toolkit evaluate_combos "$FILE" --where-to-write "$OUT/ebcc_only" --field-to-compress t \
    --l1-threshold 0.0005 --serializer-class ebcc
dc_toolkit compress "$FILE" "$OUT/ebcc_only"
```

Seven pipelines, a second or two. The parquet shows the whole trade-off at a glance:

```text
max_error_target (K)    ratio    l1_rel     keep
        8.635           65.95    0.002764   False
        2.591           35.29    0.001410   False
        0.864           18.83    0.000514   False
        0.259           12.03    0.000142   True     <- the winner
        0.0864           9.37    0.000054   True
        0.0259           6.69    0.000008   True
        0.00864          5.77    0.000004   True
```

### EBCC without any sweep

You choose the error bound yourself. EBCC's configuration is a packed list of integers that nobody should
write by hand, so let the toolkit build the pipeline:

```python
import json, os
from zarr.codecs.numcodecs import AsType
from dc_toolkit import utils

ebcc = utils.EBCC.from_params(91, 180, 0.05)     # frame height, frame width, max absolute error (here: K)
cast = AsType(encode_dtype="float32", decode_dtype="float64")     # float64 fields only; else use None
path = os.path.expanduser("~/dc_tutorial/ebcc.json")
json.dump(utils.pipeline_to_dict(None, cast, ebcc), open(path, "w"), indent=1)
```

```bash
dc_toolkit compress "$FILE" "$OUT/ebcc_manual" --vars t --pipeline "$OUT/ebcc.json" --l1-threshold 0.0005
```

```text
[compress] t: - | astype(...) | EBCC(height=91, width=180, base_cr=2, max_error_target=0.05) -> ratio=8.062 L1_rel=2.574e-05 ...
[verify-gate] t: PASS, production error norms are within the sweep thresholds.
```

The height and width are those of your (lat, lon) frame: the last two numbers of the field's shape (a side
above 2047 takes the largest divisor of it between 32 and 2047, as a sweep does). An EBCC pipeline from a sweep can
also be reused through `--pipeline path/to/manifest_t.json`, as long as the frame size is the same.

### Reading an EBCC store

An EBCC array can only be opened where `dc_toolkit[ebcc]` is installed. See section 9 if your store must
open anywhere.

---

## 9. Steering the search

**Only lossless.** `--without-lossy` keeps only pipelines that give every value back bit for bit:
compressors and lossless serializers, and for a float field no filter (`delta` rounds when it takes the
differences of floats):

```bash
mpirun -n 8 dc_toolkit evaluate_combos "$FILE" --where-to-write "$OUT/lossless" --field-to-compress t \
    --l1-threshold 0.0001 --without-lossy
```

```text
[sweep] 297 combos: 297 from the 33 x 1 x 9 grid (valid pairings) + 0 EBCC; 297 to evaluate, ~297 per node, claimed by its ranks as they free up.
best pipeline: lzma(preset=6) | - | -
Compression Ratio: 1.631 | Relative L1 Error: 0.000e+00 | Euclidean Distance: 0.000e+00
```

Zero error at a ratio of 1.6: the price of "no error at all", next to 7.6 for a largest error of 0.09 K
(section 5).

**Only some codecs.** `--compressor-class`, `--filter-class` and `--serializer-class` each restrict one stage
to one family, for example `--compressor-class zstd --serializer-class pcodec`. `none` means "leave this
stage out". For compressors and filters `all` never leaves the stage out (except `--without-lossy`, which keeps no
filter for a float field); for serializers, `all` includes plain bytes and `none` is plain bytes alone.

**Protect the physics.** Three gates go beyond average error:

| Option | Use it when |
|---|---|
| `--phys-min 0`, `--phys-max 100` | a value the pipeline moves beyond these bounds is wrong by definition (negative humidity, 101 % cloud cover); values your file already has beyond them do not count. `--phys-tolerance 0.0001` allows a hair of overshoot, as a fraction of the field's range, which lossy codecs produce on fields that sit exactly on a bound. |
| `--extremes-sensitive` | the rare large values are what matters (precipitation, gusts, CAPE): adds a gate on the top 1 % of values (of the non-zero values, for a field that is 0 almost everywhere) |
| `--gradient-gate` | differences between neighbouring cells matter (wind, pressure): adds a gate on horizontal gradients (along the other non-leading dims for a field without horizontal ones) |

**A store that opens anywhere.** Two codecs exist only inside `dc_toolkit`: EBCC, and `zfpy_flat`, a variant
of zfp that regularly wins sweeps. A store that uses either one cannot be opened by a Zarr reader without
`dc_toolkit` installed. When your data must open anywhere:

```bash
dc_toolkit compress "$FILE" "$OUT/sweep_ebcc" --stock-codecs-only
```

```text
[compress] t: the manifest best - | astype(...) | EBCC(...) needs dc_toolkit's codec entry point to be read; --stock-codecs-only takes the best stock row of results_t.parquet.
[compress] (1/1) t from results_t.parquet (stock codecs only): zstd(level=22) | quantize(digits=11, dtype=float64) | zfpy(mode=2, rate=8)
[compress] t: ... -> ratio=9.107 L1_rel=1.133e-04 ...
[verify-gate] t: PASS, production error norms are within the sweep thresholds.
```

The winner of that sweep was EBCC (section 8), so the best pipeline that needs nothing special is written
instead: a ratio of 9.1 rather than 12.0, which is what portability costs here. The sweep had evaluated
everything already, so nothing is re-run. An array that was written earlier with such a codec is rewritten.

**Back to NetCDF.** `dc_toolkit from_zarr_to_netcdf STORE.zarr --out file.nc` writes the store's fields to a
NetCDF file (an existing file needs `--overwrite`). A `compress` store holds only the fields, with their dimension names and
attributes (units) but no coordinate values, so the file has no coordinates either.

---

## 10. Your own files on a laptop

**The sample.** The sweep does not run on the whole field but on a representative sample of at most
`--eval-data-size-limit` (default `5GB`, or less after `[memcheck] auto-shrunk`): whole horizontal fields at
time steps and levels spread evenly through the file. A field that fits is evaluated in full. A sample keeps
at least 3 time steps and 3 levels (all, where the field has fewer; ensemble members and forecast steps count
as time steps), so a smaller limit is raised to that
(`[sample] raised the sample budget ...`); a field with no time or vertical dimension to thin is its own
sample. For big files on a laptop, lower the limit, for example `--eval-data-size-limit 512MiB`.
[SAMPLING.md](SAMPLING.md) explains the rules.

A sample must show how the field varies. When it holds a single value (or none that is finite) while the
field varies, the sweep skips the field (`[var] skipping ...`, an error for the `--field-to-compress`): raise
`--eval-data-size-limit`. A field whose finite values are all one number, or that has none, is stored
losslessly with Zstd, without a search.

**Memory.** The processes of one machine share a single copy of the sample; each adds a working set of about
twice the sample for the pipeline it is evaluating, so memory grows with the number of ranks. The `[memory]`
line at the start of a sweep shows the estimate. If it does not fit, the toolkit shrinks the sample by itself
(`[memcheck] auto-shrunk ...`), down to the minimum above, or refuses to start and tells you what to change.
The two knobs are `--eval-data-size-limit` and the number of ranks you start (`mpirun -n`). The estimate is
checked against the machine's total memory, not what is free at that moment; only the sample itself (twice the
sample on the first rank) is checked against what is free, and the sweep refuses with `[memcheck] REFUSING to
proceed` when it does not fit. When other programs hold much of the memory, close them or start fewer ranks,
or the sweep will swap.

**Time.** A sweep's duration grows with the sample size and with the number of pipelines. Start with
`--max-evals` to see whether the numbers make sense, then run the full sweep. Leave your laptop usable by
starting fewer ranks, for example `mpirun -n 4`.

**Interruptions.** Pressing Ctrl-C loses only the pipelines being evaluated at that moment: run the same
command again and the sweep continues where it stopped (`[resume] N of M combo(s) ... already recorded`). It
starts over only when something that affects the measurements changed: the file, the sample (its size, its
sampling settings or its values), the chunk settings, the physical bounds, the toolkit's measuring code, or the
versions of the compression libraries (EBCC's tuning environment variables included); the previous results
are then kept as `*.previous`. A pipeline that crashes the whole program (rare) is tried alone at the end of
the next run, and left out if it crashes again. An auto-shrunk sample is sized for
the number of ranks, so resume with the same `mpirun -n`: a different count can build a different sample (when
the field is larger than the budget), which starts the field over. `--no-resume` forces a fresh start.

**Fields with NaN.** Cells that are NaN in your file (fill values, masked land or sea) are left out of every
error norm, and a pipeline must give them back as NaN. The sweep counts them over the whole field first; when
there are any, it leaves out the codecs that would turn them into numbers (`fixedscaleoffset`, `zfpy`, `delta`
on floats, EBCC). In section 5's check, use `np.nanmax`.

**`compress` is safe to repeat.** A field the store already holds as the run would write it is skipped; a
field that failed, or whose pipeline, budget or source file changed, is written again.

---

## 11. Troubleshooting

| You see | It means | Do this |
|---|---|---|
| `[oversubscription-check] ... Aborting` | the thread variables are not pinned | run the `export` line of section 1.4 |
| the install script stops at `mpicc` | no MPI library | install one (section 1.1), then run the install script again |
| `Unsupported file format` | the extension is not `.nc`, `.grib` or `.zarr` | rename the file; the format is chosen from the extension |
| `Field x not found in dataset. Available fields: [...]` | a typo in `--field-to-compress` | pick a name from the list it prints |
| `--with-ebcc needs the ebcc package` | EBCC is not installed | `pip install -e ".[ebcc]"` (section 8) |
| `--serializer-class ebcc has nothing for this field: ...` | the field is not a float (lat, lon) grid, or no tile fits its frame (section 8) | sweep it without EBCC |
| `invalid pipeline ... (e.g. FixedScaleOffset->ZFPY, ...)` | one of the pairings section 7 lists as refused | fix that pairing: `bitround` before `zfpy` only with the full mantissa, no `fixedscaleoffset` before `zfpy` (or before 8-bit `pcodec`), EBCC with no compressor and at most the `astype` filter |
| `[verify-gate] FAIL` and exit status 1 | the written field exceeds the budget, or its pipeline changed missing values (`pass_finite`) or moved values beyond `--phys-min`/`--phys-max` (`pass_bounds`) | loosen the budget, or choose a more careful pipeline (for `pass_finite`, one without `fixedscaleoffset` and `zfpy`); nothing was left in the store |
| `[verify-gate] ... no thresholds ...: the error norms are advisory` | `--pipeline` without a budget | add `--l1-threshold` if you want the norms enforced |
| `t already in ... as requested; skipping.` | the store already holds the field as this run would write it | add `--no-skip-existing` to write it anyway |
| `... does not match sweep_state_t.json ...` | a later sweep of the field has not finished | run that `evaluate_combos` again; `compress` uses only the last sweep's winner |
| `another sweep of t is writing into ...` | a sweep of the same field into the same directory is running | wait for it; if none runs, remove the lock file the message names |
| `[resume] ... starting this field from scratch` | the file, sample, chunking, bounds, measuring code or library versions changed | nothing; the recorded results do not match, so the field is measured again |
| `[memcheck] REFUSING ...` | too little memory is free at that moment: for a sweep, for the sample (twice it on the first rank); for `compress`, for one block of the write | close other programs, or lower `--eval-data-size-limit` (sweep) or `--shard-mib` (compress) |
| `[memcheck] FATAL: the smallest sample of 'x' ... does not fit` | the field's smallest sample (3 time steps × 3 levels) and the ranks' working sets need more memory than the machine has | start fewer ranks (`mpirun -n`) |
| `[sample] raised the sample budget ...` | the limit is below the field's smallest sample | nothing: a smaller sample would not show the field's time steps and levels |
| `[var] skipping x: the sample holds the single value ...` | the time steps and levels the sample keeps do not show how the field varies | raise `--eval-data-size-limit` |
| `[shared-sample] FATAL ... Allocate_shared failed` | the shared memory is smaller than the sample, usually in a container | `docker run --shm-size` of at least the sample size |
| `[cr-drift] WARNING` | the sample predicted a different ratio than the whole field achieved | informative; a larger sample predicts better |
| `[var] skipping grid geometry (CF bounds): ...` | helper variables were left out on purpose | nothing; name one with `--field-to-compress` if you really want it |
| the largest error is bigger than you expected | the budget is relative to the field's magnitude | section 5 |

---

## 12. Cheat sheet

```bash
# every new terminal
source venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1 \
       NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1

# search, then write
mpirun -n 8 dc_toolkit evaluate_combos FILE --where-to-write DIR --field-to-compress VAR --l1-threshold 0.0005
dc_toolkit compress FILE DIR

# a quick look first                      ... --max-evals 300
# every field of the file                 (leave out --field-to-compress)
# a new budget, no new measurements       (run evaluate_combos again with another --l1-threshold)
# overwrite a field in the store          dc_toolkit compress FILE DIR --no-skip-existing

# no sweep: your own pipeline
dc_toolkit compress FILE DIR --vars VAR --pipeline pipeline.json --l1-threshold 0.0005
dc_toolkit compress FILE DIR --vars VAR --pipeline other_sweep/manifest_VAR.json --l1-threshold 0.0005

# EBCC
mpirun -n 8 dc_toolkit evaluate_combos FILE --where-to-write DIR --field-to-compress VAR --l1-threshold 0.0005 --with-ebcc
mpirun -n 8 dc_toolkit evaluate_combos FILE --where-to-write DIR --field-to-compress VAR --l1-threshold 0.0005 --serializer-class ebcc

# lossless only                           ... --without-lossy
# a store any Zarr reader opens           dc_toolkit compress FILE DIR --stock-codecs-only
# look inside a store                     dc_toolkit open_zarr_and_inspect STORE.zarr
# back to NetCDF                          dc_toolkit from_zarr_to_netcdf STORE.zarr --out file.nc
```

Where to go next: the [README](../README.md) documents every output file and the cluster setup, and
[PARALLELIZATION.md](PARALLELIZATION.md) explains how the commands use your cores, and [SAMPLING.md](SAMPLING.md)
how the sweep samples a field. Both commands are also available from a graphical interface
(`dc_toolkit run_web_ui`).
