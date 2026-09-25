# Sampling in `evaluate_combos`

`evaluate_combos` scores thousands of compression pipelines, and it does not run them on the whole field: a rank
holds about two copies of what it evaluates, so a 300 GiB field would not fit in a node's memory even once, and
thousands of round trips of it would take days. It runs them on a **sample**, and `compress` then writes the winner
and checks it on the whole field. This document explains how the sample is built, sized and used, what it cannot
see, and which data layouts the sampler does not handle yet.

The code: `minimum_sample`, `build_representative_sample` and `_allocate_stride_plan` in `src/dc_toolkit/utils.py`
(section 2), `sweep_sample_limit` and `sweep_build_sample` in `src/dc_toolkit/utils_cli.py` (section 5). How the
ranks of a node share one copy of the sample is in [PARALLELIZATION.md](PARALLELIZATION.md).

---

## What a sample is

A sample is a set of **whole horizontal fields**, every cell of the global or regional grid, taken at a few time
steps and a few levels. Only the time and vertical dimensions are thinned. The horizontal dimensions are never cut:
the codecs exploit spatial structure (zfp's blocks, shuffles and deltas along neighbouring cells), and a thinned
horizontal grid would not show it.

One horizontal field at one time step and one level is a **slab**. On the native R02B10 grid a float32 slab is
83,886,080 cells × 4 bytes = 320 MiB. A sample is a number of slabs.

## How the sample is built

### 1. The dimensions are sorted into three kinds

| Kind | Recognised by (CF metadata first, then the name) | Treatment |
|---|---|---|
| time-like | units `<unit> since <date>`, `axis: T`, a `calendar` attribute, standard name `time`, `forecast_reference_time`, `forecast_period` or `realization`; names such as `time`, `*_time`, `t`, `step`, `member`, `member_id`, `ensemble`, `realization` | thinned |
| vertical | `axis: Z`, `positive: up` or `down`, standard names such as `height`, `air_pressure`, `depth`, `model_level_number`; names such as `lev`, `plev`, `level`, `height`, `depth`, `ilev`, `bottom_top` | thinned |
| horizontal | everything else: `ncells`, `lat`/`lon`, `x`/`y`, ... | kept whole |

Ensemble members and forecast lead times count as time-like: they are thinned together with time (see
[below](#data-layouts-the-sampler-does-not-handle-yet)).

### 2. The budget

`sweep_sample_limit` decides how many bytes the sample may use, in this order:

1. It starts from `--eval-data-size-limit` (default `5GB`; `santis.run` passes each field's `SAMPLE`, `2GiB` unless
   the entry names another).
2. A field that fits in the budget is its own sample: nothing is thinned (`[sample] field fits under limit`).
3. The budget is raised to the field's **smallest sample**: 3 time steps × 3 levels, or all of them where the field
   has fewer (`[sample] raised the sample budget ...`). A field without time and vertical dimensions cannot be
   thinned; its smallest sample is the whole field.
4. The budget is capped by the node's memory. A node holds one shared copy of the sample, and each rank holds about
   twice the sample while it evaluates a pipeline (the decoded copy and the encoded bytes), plus two inner chunks:

   ```
   S + R × 2 × S + R × 2 × chunk  ≤  --memory-threshold × node memory      (S = sample, R = ranks per node)
   ```

   On a Santis node (849.6 GiB for the job, threshold 0.8, 16 MiB chunks) the sample is capped at 20.6 GiB with 16
   ranks per node, 10.4 GiB with 32 and 5.3 GiB with 64. A larger budget is shrunk (`[memcheck] auto-shrunk ...`).
   When even the smallest sample does not fit, the sweep stops (`[memcheck] FATAL: the smallest sample of ...`)
   rather than take less. With several nodes, all use the smallest node's value.

### 3. The plan: how many time steps and levels

The budget divided by the slab size is the number of slabs the sample may hold. `--sampling-policy` splits it
between the time and the vertical dimensions:

- **cascade** (default) favours levels. It first keeps a floor of levels, `--vertical-floor` (by default
  max(4, ⌈log₂ levels⌉), so 7 of 120 levels), lowered until 3 time steps fit beside it but never below 3. Then it
  spends the budget on time steps, up to all of them, and gives what is left back to levels.
- **balanced** splits the budget evenly between the two, roughly the square root each.

Both keep at least 3 time steps and 3 levels. For the native R02B10 `qc` field (time 8 × height 120, 300 GiB):

| Budget | Slabs | cascade (time × levels) | balanced |
|---|---|---|---|
| 1 or 2 GiB, raised to 2.8 GiB | 9 | 3 × 3 | 3 × 3 |
| 5 GiB | 16 | 3 × 5 = 4.7 GiB | 4 × 4 = 5.0 GiB |
| 10 GiB | 32 | 4 × 8 = 10.0 GiB | 5 × 6 = 9.4 GiB |
| 30 GiB | 96 | 8 × 12 | 8 × 12 |

The 30 GiB row needs 10 ranks per node or fewer (the cap of step 2). A 2-D field has no levels to split: the R02B10
`tot_prec` (96 quarter-hourly steps, 30 GiB) keeps 3 time steps at 1 GiB and 6 at 2 GiB.

### 4. Which indices

Each thinned dimension is cut into as many equal blocks as indices are kept, and the middle of each block is taken:

```
index_k = floor((k + 1/2) × size / n)        k = 0 … n−1
```

3 of 8 time steps: blocks [0, 2.67), [2.67, 5.33), [5.33, 8), indices 1, 4, 6 (03:00, 12:00 and 18:00 UTC in a
three-hourly day). 3 of 120 levels: 20, 60, 100. The two ends of a dimension, the model top and the first time step,
are the least typical, and are only kept when many indices are: 5 of 8 time steps gives 0, 2, 4, 5, 7.

The log line says what was kept:

```
[sample] field is 300.0 GiB > limit 2.8 GiB; policy=cascade; strided time=3/8 [1, 4, 6], height=3/120 [20, 60, 100] | preserved spatial: ncells -> 2.8 GiB.
```

### 5. Read, share, scan

Rank 0 reads only the chosen slabs and puts them in shared memory, one copy per node. Meanwhile all ranks read the
whole field once, block by block, for its finite minimum and maximum. That full-field range serves three purposes:

- FixedScaleOffset's scale and offset. The filter does not clip, so it is fitted to the whole field's range, never to
  the sample's; without a range (a failed read) FixedScaleOffset is left out.
- `--phys-tolerance`, which is a fraction of the field's range.
- Judging a sample without variation:
  - a field whose finite values are all one number, or that has none, needs no search: it is stored losslessly with
    Zstd;
  - a sample with a single value, or no finite value, while the field varies says nothing about the codecs: the
    field is skipped (`[var] skipping ...`, an error when it was named with `--field-to-compress`). A larger
    `--eval-data-size-limit` lets the sample reach the variation.

Rank 0 also computes, once, what the gates take from the whole sample: the q99 cut of `--extremes-sensitive`, and
whether the sample is finite (EBCC is left out when it is not).

### 6. Use

Every pipeline compresses the whole sample in memory, chunked with the rule `compress` applies to the whole field
(`--inner-chunk-mib`, `--spatial-split`), decodes it, and the ratio and the errors are measured on the sample. The
gates decide from those numbers. `compress` then writes the whole field with the winner and measures again: the
verify gate re-checks the errors, and the CR-drift check compares the ratio with the sweep's.

### 7. Resume

`sweep_state_{var}.json` records what the recorded rows were measured on, the sample included: its shape, dtype and a
digest of its bytes, the sampling policy and vertical floor, and the full-field range. A change starts the field
over. The budget depends on the ranks per node through the memory cap, so resuming a large field with another rank
count can give another sample, and restart the field.

## What a sample cannot see

These limits apply to the data the toolkit handles today.

**Level index is not height.** Indices are spread evenly over the levels, and model levels are not spread evenly
over the atmosphere: ICON crowds them towards the ground. Mean geopotential height of native R02B10 levels at one
time step:

| index | 0 | 20 | 60 | 82 | 100 | 108 | 119 |
|---|---|---|---|---|---|---|---|
| height | 84 km | 56 km | 23 km | 11 km | 4.1 km | 2.0 km | 0.26 km |

Only the lowest third of the levels (index 80 and above) lies below about 12 km. The 3 × 3 minimum keeps one level
there (100); for cloud and precipitation fields the other two are mostly zeros. With the default cascade policy, a
5 GiB budget keeps levels 84 and 108 (10 and 2 km) among its five, and 10 GiB keeps 82, 97 and 112 (11, 5 and
1.2 km) among its eight.

To address: choose levels evenly in pressure (mass) or height rather than in index, or only where the field varies,
or let a field entry restrict the levels (to the troposphere, say).

**Sample chunks are not production chunks.** For a native 3-D field the sample's 16 MiB inner chunks hold 1 time
step × 3 levels × 1.4 M cells, while the production chunks hold 1 time step × 120 adjacent levels × 35 k cells.
Codecs that use the correlation between levels (zfp's 3-D blocks) see levels about 40 apart in the sample and
neighbours in production, so their ratio on the sample is probably pessimistic; this has not been measured. Fields
without a vertical dimension have the same chunks in both.

To address: keep a few adjacent levels at each chosen index, so that the sample's chunks look like production
chunks.

**One file.** The sample comes from the file the sweep is given, so other days and seasons are not seen. Every slab
is global, so one UTC time already covers every local time of day; more time steps add weather variety.

**Missing values outside the sample.** The sweep drops a pipeline that turns the sample's NaN into numbers
(FixedScaleOffset, zfp) or numbers into NaN. When a field's NaN lie only outside the sample, such a pipeline can win
the sweep; `compress` then refuses it (`pass_finite`) and writes nothing.

To address: count the non-finite cells during the full-field range pass, which reads every block already, and leave
out the pipelines that cannot keep NaN when the field has any.

## Data layouts the sampler does not handle yet

DYAMOND output and the bundled TIGGE file have at most one time dimension, one vertical dimension and the horizontal
ones. Fields may lack the time dimension (static fields), the vertical one (surface fields such as `tot_prec`), or
carry a vertical dimension of size 1; the sampler handles all of those. Other archives add dimensions, and the
following cases are not handled deliberately yet.

**Ensembles stored in one file.** An ensemble runs the same model many times from slightly different starting
conditions; each run is a *member*. Weather centres (ECMWF's ensemble read from GRIB: `number`, standard name
`realization`) and CMIP6 collections (`member_id`) can stack the members along a dimension. That dimension is
time-like, and the 3-index minimum is shared by all the time-like dimensions: for (member 50, time 10, level 20) the
smallest sample is 3 members × 1 time step × 3 levels.

To address: give members their own minimum, e.g. 3 members × 3 time steps × 3 levels = 27 slabs, three times the
memory; or decide whether members need sampling at all, since the members of an ensemble are statistically alike and
a few members with more time steps may represent the field better.

**Forecasts: start time × lead time.** A forecast archive keeps the start of each forecast (`time`,
`forecast_reference_time`) and the lead time after it (`step`, `forecast_period`); a value describes the moment
start + lead time. Both dimensions are time-like and share the minimum, so the smallest sample holds three lead times
of one forecast, or one lead time of three forecasts, whichever dimension is longer. The lead time changes a field
more than the start does: accumulated fields such as precipitation are zero everywhere at lead time 0.

To address: count start times and lead times separately, e.g. early, middle and late lead times of a few starts.

**Dimensions that are neither time, vertical nor horizontal.** Spectral bands, land-surface tiles, percentiles of a
probabilistic forecast: a dimension not recognised as time-like or vertical is treated as horizontal. It is never
thinned, so the sample grows (the memory cap still applies, and the sweep stops when the smallest sample does not
fit), and the gradient gate differences along it as if its entries were neighbouring cells.

To address: a fourth kind of dimension, kept whole or thinned like time but never differenced, and a way to name the
kind of a dimension on the command line when its metadata does not say.

**Time or vertical dimensions the rules do not recognise.** A dimension without CF metadata and with a name the rules
do not list (`date` or `init`, say) is treated as horizontal as well, with the same consequences.

To address: the same command-line naming, and extending the name lists as such files come up.

**Large fields without time and vertical dimensions.** Such a field is its own smallest sample. A static field such
as the orography or the land–sea mask costs little; a field larger than the memory cap stops the sweep.

To address, if such fields appear: sample contiguous blocks of the horizontal grid, which keeps the local structure
the codecs use.

**Several files.** The sweep reads one file. Members, models or days stored in separate files, like the DYAMOND3
multi-model collection with one model per file, are swept file by file, and the sample of one file does not see the
others.

To address: accept several files as one dataset and sample across them.
