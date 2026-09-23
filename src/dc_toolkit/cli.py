"""
dc_toolkit command-line interface: the commands and their options only.
The work happens in utils_cli.py (command helpers) and utils.py (library).

Pipeline
  evaluate_combos   sweep compressor x filter x serializer on a sample
                    -> results_{var}.parquet and manifest_{var}.json (best pipeline)
  compress          persist every swept field (or --vars, or a --pipeline of your
                    own) into {dataset}.zarr and consolidate the store's metadata

Sections
  1. Shared options
  2. Pipeline commands
  3. Store utilities & format conversion
  4. Analysis & plotting
  5. UIs & help
"""
import os
import subprocess
import sys
import warnings
from pathlib import Path

import click
import dask

from dc_toolkit import utils, utils_cli

warnings.filterwarnings("ignore", message="Numcodecs codecs are not in the Zarr version 3 specification.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part in the Zarr format 3.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message="Engine 'cfgrib' loading failed", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in square")
warnings.filterwarnings("ignore", message=r".*leaked semaphore objects.*", category=UserWarning,
                        module=r"multiprocessing\.resource_tracker")


@click.group()
def cli():
    pass


# =============================================================================
# 1. SHARED OPTIONS
# =============================================================================

_CODEC_SPACE_OPTIONS = [
    click.option("--compressor-class", default="all", show_default=True,
                 type=click.Choice(["all", "none", "blosc", "lz4", "zstd", "zlib", "bz2", "lzma"], case_sensitive=False),
                 help="Restrict the compressors to one class; 'none' = no compressor."),
    click.option("--filter-class", default="all", show_default=True,
                 type=click.Choice(["all", "none", "delta", "bitround", "quantize", "fixedscaleoffset", "astype"],
                                   case_sensitive=False),
                 help="Restrict the filters to one class; 'none' = no filter.  Integer fields have Delta as "
                      "their only filter and fall back to it, with a message, for any class but 'none'; a float "
                      "field without the class is skipped (an error when it is the --field-to-compress)."),
    click.option("--serializer-class", default="all", show_default=True,
                 type=click.Choice(["all", "none", "pcodec", "zfpy", "ebcc"], case_sensitive=False),
                 help="Restrict the serializers to one class; 'all' includes plain bytes, 'none' is plain bytes "
                      "alone.  Fields the class cannot take are skipped, as for --filter-class."),
    click.option("--with-lossy/--without-lossy", default=True, show_default=True,
                 help="Include lossy filters and serializers in the codec space."),
    click.option("--with-ebcc/--without-ebcc", default=False, show_default=True,
                 help="Add the EBCC serializer (lossy; needs --with-lossy and the optional ebcc package): "
                      "float (lat, lon) frames only, no compressor and no filter except the AsType "
                      "down-cast, encodes at ~1-2 MB/s per core."),
]
_CHUNK_OPTIONS = [
    click.option("--inner-chunk-mib", type=click.IntRange(min=1), default=16, show_default=True,
                 help="Target zarr chunk size in MiB (compress reuses the sweep's value)."),
    click.option("--max-inner-chunk-mib", type=click.IntRange(min=1), default=256, show_default=True,
                 help="Warn when --no-spatial-split produces a chunk above this size (MiB)."),
    click.option("--spatial-split/--no-spatial-split", default=True, show_default=True,
                 help="Split spatial dims (horizontal first, vertical last) when one timestep exceeds "
                      "--inner-chunk-mib.  --no-spatial-split keeps one full timestep per chunk instead "
                      "(see --max-inner-chunk-mib); compress reuses the sweep's setting."),
]
_CHUNK_OVERRIDE_OPTIONS = [  # compress: None means "the sweep's value from the manifest"
    click.option("--inner-chunk-mib", type=click.IntRange(min=1), default=None,
                 help="Target zarr chunk size in MiB (default: the sweep's value, else 16)."),
    click.option("--max-inner-chunk-mib", type=click.IntRange(min=1), default=None,
                 help="Warn above this inner chunk size (MiB) with --no-spatial-split (default: the sweep's, else 256)."),
    click.option("--spatial-split/--no-spatial-split", default=None,
                 help="Split spatial dims when one timestep exceeds --inner-chunk-mib (default: the sweep's, else on)."),
]
_OVERSUBSCRIPTION_OPTION = click.option(
    "--oversubscription-check/--no-oversubscription-check", default=True, show_default=True,
    help="Abort at startup unless " + ", ".join(utils.THREAD_ENV_VARS) + " are all set to 1.")
_MEMORY_OPTION = click.option(
    "--memory-threshold", type=click.FloatRange(0.05, 0.95), default=0.80, show_default=True,
    help="Max fraction of memory an estimated footprint may use.  evaluate_combos shrinks its sample until "
         "one node's footprint fits this fraction of the node's budget (cgroup limit, else RAM) and aborts "
         "when nothing fits; compress refuses a write whose peak does not fit the available RAM.")
_PERSIST_OPTIONS = _CHUNK_OVERRIDE_OPTIONS + [
    click.option("--shard-mib", type=click.IntRange(min=1), default=512, show_default=True,
                 help="Target shard size in MiB (an integer number of inner chunks).  Sharding is "
                      "skipped when a shard would hold fewer than two chunks."),
    click.option("--threads", type=click.IntRange(min=1), default=None,
                 help="Dask workers for the write (default: visible cores).  Peak memory ~ threads x "
                      "(max(source block, shard) + 3 x shard); the memory guard refuses what does not fit."),
    _OVERSUBSCRIPTION_OPTION, _MEMORY_OPTION,
]


def _finite(ctx, param, value):
    """Refuse NaN and +-inf: a non-finite bound or budget silently rejects or passes every combo."""
    if value is not None and (value != value or abs(value) == float("inf")):
        raise click.BadParameter("must be finite")
    return value


_VERIFY_OPTIONS = [
    click.option("--verify/--no-verify", default=True, show_default=True,
                 help="Re-read the store after writing and recompute the error norms "
                      "(roughly doubles wall time; skip for trusted re-runs)."),
    click.option("--verify-gate/--no-verify-gate", default=True, show_default=True,
                 help="With --verify, fail a field whose production norms exceed the sweep thresholds "
                      "or physical bounds in manifest_{var}.json (the gradient gate is sweep-only).  "
                      "--no-verify-gate only warns."),
]


# =============================================================================
# 2. PIPELINE COMMANDS
# =============================================================================

@cli.command("evaluate_combos")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--where-to-write", "where_to_write", required=True,
              type=click.Path(dir_okay=True, file_okay=False, exists=False),
              help="Output directory (config_space_{var}.csv, per-rank CSVs, results_{var}.parquet, "
                   "manifest_{var}.json).  Created if missing.")
@click.option("--field-to-compress", default=None,
              help="Field to sweep (default: every non-empty integer/float32/float64 variable with at least "
                   "one dim, CF bounds excepted).")
@click.option("--eval-data-size-limit", default="5GB", callback=utils_cli.size_option_callback, show_default=True,
              help="Budget of the representative sample the combos are scored on (e.g. 5GB, 512MiB).")
@_OVERSUBSCRIPTION_OPTION
@utils_cli.add_options(_CHUNK_OPTIONS)
@_MEMORY_OPTION
@click.option("--l1-threshold", type=click.FloatRange(min=0.0), required=True, callback=_finite,
              help="Relative L1 error budget (e.g. 0.005 = 0.5%).  The anchor for the other gates.")
@click.option("--l2-threshold", type=click.FloatRange(min=0.0), default=None, callback=_finite,
              help="Relative L2 budget (default: 2 x L1).")
@click.option("--linf-threshold", type=click.FloatRange(min=0.0), default=None, callback=_finite,
              help="Relative Linf (worst cell) budget (default: 10 x L1).")
@click.option("--bias-threshold", type=click.FloatRange(min=0.0), default=None, callback=_finite,
              help="Relative bias budget |mean signed error| / mean|orig| (default: 0.5 x L1).")
@click.option("--q99-threshold", type=click.FloatRange(min=0.0), default=None, callback=_finite,
              help="Relative budget over cells with |value| >= the 99th percentile (default: 2 x L1).  "
                   "Only with --extremes-sensitive.")
@click.option("--l2-gate/--no-l2-gate", default=True, show_default=True, help="Enable the L2 gate.")
@click.option("--linf-gate/--no-linf-gate", default=True, show_default=True, help="Enable the Linf gate.")
@click.option("--bias-gate/--no-bias-gate", default=True, show_default=True, help="Enable the bias gate.")
@click.option("--extremes-sensitive/--no-extremes-sensitive", default=False, show_default=True,
              help="Enable the q99 extreme-tail gate (precip, gusts, CAPE, radiation peaks).")
@click.option("--phys-min", type=float, default=None, callback=_finite, help="Reject combos whose decoded sample dips below this.")
@click.option("--phys-max", type=float, default=None, callback=_finite, help="Reject combos whose decoded sample exceeds this.")
@click.option("--phys-tolerance", type=click.FloatRange(0.0, 1.0), default=0.0, show_default=True, callback=_finite,
              help="Slack for --phys-min/--phys-max as a fraction of the field's value range: a lossy codec "
                   "rings past a bound the field sits on by a hair. Stored in the manifest as an absolute "
                   "value, so compress's verify gate applies the same slack.")
@click.option("--gradient-gate/--no-gradient-gate", default=False, show_default=True,
              help="Enable the spatial-gradient gate (one more pass over the sample per combo; "
                   "for winds, pressure).")
@click.option("--gradient-threshold", type=click.FloatRange(min=0.0), default=0.1, show_default=True, callback=_finite,
              help="Max relative L1 error of the finite-difference field (absolute fraction, not x L1).")
@click.option("--gradient-shortcircuit/--no-gradient-shortcircuit", default=True, show_default=True,
              help="Only compute the gradient for combos that already pass the cheap gates.")
@utils_cli.add_options(_CODEC_SPACE_OPTIONS)
@click.option("--sampling-policy", type=click.Choice(["cascade", "balanced"]), default="cascade",
              show_default=True,
              help="How an over-budget field is thinned (only its time and vertical dims; the others are kept "
                   "whole): 'cascade' spends the budget on time steps first, keeping a minimum of vertical "
                   "levels; 'balanced' splits it evenly across the time and vertical dims.")
@click.option("--vertical-floor", type=click.IntRange(min=1), default=None,
              help="Minimum vertical levels kept by the cascade policy "
                   "(default: max(4, ceil(log2(n_levels)))).")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip combos already recorded in config_space_{var}_rank*.csv: their metrics are reused "
                   "and the gates re-applied with the current thresholds.  A changed file, sample, chunk "
                   "setting or library version (recorded in sweep_state_{var}.json) restarts the field.")
@click.option("--max-evals", type=click.IntRange(min=1), default=None,
              help="Cap the Cartesian product (quick test runs); EBCC combos are always included.")
@click.pass_context
def evaluate_combos(ctx, **_):
    """
    Sweep compressor x filter x serializer combinations on a representative
    sample of each field, gate them on error thresholds, and record every
    result in results_{var}.parquet and the best pipeline in manifest_{var}.json.

    \b
    Parallelism: MPI ranks split the config space and evaluate one pipeline at
    a time each; the ranks of a node share one copy of the sample.  Launch one
    rank per core:
      srun --nodes=N --ntasks-per-node=32 --cpus-per-task=1 dc_toolkit evaluate_combos ...
      mpirun -n 8 dc_toolkit evaluate_combos ...   (a laptop)
    Everything runs in memory; use `compress` to write the winners.
    """
    opts = utils_cli.opts(ctx)
    sweep = utils_cli.sweep_setup(opts)
    # array.chunk-size must be set before any open() with chunks="auto".
    with dask.config.set({"array.chunk-size": "512MiB", "scheduler": "synchronous"}):  # one core per rank
        ds = utils.open_dataset(opts.dataset_file, opts.field_to_compress, rank=sweep.rank)
        variables = utils_cli.sweep_variables(ds, opts.field_to_compress, sweep.rank)
        for var in variables:
            utils_cli.sweep_variable(ds[var], var, opts, sweep, n_vars=len(variables))


_VERIFY_THRESHOLD_OPTIONS = [
    click.option(f"--{k}-threshold", type=click.FloatRange(min=0.0), default=None, callback=_finite,
                 help=f"Relative {label} budget for the verify gate, overriding the sweep's value in "
                      f"manifest_{{var}}.json (the only way to gate a --pipeline field without a manifest).")
    for k, label in (("l1", "L1"), ("l2", "L2"), ("linf", "Linf"), ("bias", "bias"))
]


@cli.command("compress")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.option("--vars", "vars_filter", default=None,
              help="Comma-separated fields to write (default: every field with a manifest_{var}.json "
                   "or results_{var}.parquet in WHERE_TO_WRITE).")
@click.option("--pipeline", default=None,
              help="Write the --vars fields with this pipeline instead of the sweep's best: a JSON object "
                   "with compressor, filter and serializer, or the path of a file holding one; a "
                   "manifest_{var}.json works directly.")
@click.option("--stock-codecs-only", is_flag=True, default=False,
              help="Write only pipelines a zarr client without dc_toolkit can decode: when the sweep's best uses a codec "
                   "that needs dc_toolkit's zarr.codecs entry point (numcodecs.zfpy_flat, numcodecs.ebcc_filter), "
                   "take the best stock row of results_{var}.parquet instead.  A --pipeline with such a codec "
                   "is refused, a field already stored with one is rewritten even under --skip-existing, and "
                   "the run fails while the store still holds any such array.")
@utils_cli.add_options(_PERSIST_OPTIONS)
@utils_cli.add_options(_VERIFY_OPTIONS)
@utils_cli.add_options(_VERIFY_THRESHOLD_OPTIONS)
@click.option("--cr-drift-tol", type=click.FloatRange(0.0, 10.0), default=0.25, show_default=True,
              help="Allowed fractional drift between the achieved and the sweep's compression ratio.")
@click.option("--cr-drift-gate/--no-cr-drift-gate", default=False, show_default=True,
              help="Fail a field whose ratio falls short of the sweep's by more than --cr-drift-tol "
                   "(default: warn only).")
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True,
              help="Skip fields already present in the store.  A field only appears there once its write "
                   "and gates succeeded, so failed or interrupted fields are retried.")
@click.option("--continue-on-error/--no-continue-on-error", default=True, show_default=True,
              help="Log and go on when a field fails (default) instead of stopping at the first failure.  "
                   "The exit status is 1 either way when any field failed.")
@click.option("--consolidate/--no-consolidate", default=True, show_default=True,
              help="Consolidate the store's metadata at the end so readers open it quickly.  "
                   "--no-consolidate drops any earlier consolidated metadata instead, since this run "
                   "would make it stale.")
@click.pass_context
def compress(ctx, **_):
    """
    Persist fields into {WHERE_TO_WRITE}/{dataset}.zarr, one zarr array per
    field, with the pipeline evaluate_combos found best for each of them (read
    from manifest_{var}.json, else the best kept row of results_{var}.parquet),
    or with the --pipeline you pass.  A field is written under a staging name,
    re-read and gated, and only then put in place, so the store never holds a
    failed or half-written field.  The store's metadata is consolidated at the
    end, batch_manifest.json records every field, and the exit status is 1
    when any field failed.  Single process; dask threads parallelise each write.
    """
    opts = utils_cli.opts(ctx)
    utils_cli.require_single_process("compress")
    click.echo(utils_cli.version_banner("compress"))
    utils_cli.single_process_setup(opts)
    os.makedirs(opts.where_to_write, exist_ok=True)
    candidates, manifests, dropped = utils_cli.compress_candidates(opts)
    if not candidates and not dropped:
        raise click.ClickException("no variables to compress.  Did evaluate_combos run against the same directory?")
    merged_path = utils_cli.merged_store_path(opts.where_to_write, opts.dataset_file)
    if Path(merged_path).resolve() == Path(opts.dataset_file).resolve():
        raise click.ClickException(f"the output store {merged_path} is the input dataset; pick another "
                                   f"WHERE_TO_WRITE.")
    ds = utils.open_dataset(opts.dataset_file)
    utils_cli.remove_staged(merged_path)  # leftovers of an interrupted run
    existing = utils_cli.existing_arrays(merged_path)

    results = {var: {"status": "no-pipeline", "reason": reason} for var, reason in dropped.items()}
    any_error, stopped_at = bool(dropped), None
    with dask.config.set(scheduler="threads", num_workers=opts.threads):
        for i, cand in enumerate(candidates, start=1):
            var = cand["var"]
            click.echo(f"\n[compress] ({i}/{len(candidates)}) {var} from {cand['source']}: {cand['name']}")
            if opts.skip_existing and var in existing:
                if not opts.stock_codecs_only or utils_cli.array_is_stock(merged_path, var):
                    click.echo(f"[compress] {var} already in {merged_path}; skipping.")
                    results[var] = {"status": "skipped-existing"}
                    continue
                click.echo(f"[compress] {var} in {merged_path} needs dc_toolkit to be read; rewriting it.")
            if var not in ds.data_vars:
                any_error = True
                click.echo(f"[compress] ERROR: variable '{var}' not in dataset; skipping.")
                results[var] = {"status": "missing-from-dataset"}
            else:
                try:
                    results[var] = utils_cli.compress_one(ds[var], var, cand, manifests.get(var), merged_path, opts)
                except (Exception, SystemExit) as e:  # SystemExit: a memory guard refused the write
                    any_error = True
                    message = (e.message if isinstance(e, click.ClickException)
                               else "refused by a guard (see above)" if isinstance(e, SystemExit) else repr(e))
                    click.echo(f"[compress] ERROR on {var}: {message}")
                    results[var] = {"status": "error", "error": message}
                    if not opts.continue_on_error:
                        click.echo("[compress] stopping at the first failure (--no-continue-on-error).")
                        stopped_at = i
                        break

    for cand in candidates[stopped_at:] if stopped_at else []:
        results.setdefault(cand["var"], {"status": "not-attempted", "reason": "the run stopped at an earlier failure"})
    if opts.stock_codecs_only:
        left = sorted(v for v in utils_cli.existing_arrays(merged_path) if not utils_cli.array_is_stock(merged_path, v))
        if left:
            any_error = True
            click.echo(f"[compress] ERROR: {merged_path} still holds array(s) that need dc_toolkit to be read: "
                       f"{', '.join(left)}.")
    if Path(merged_path).is_dir():
        if opts.consolidate:
            names = utils_cli.consolidate_store(merged_path)
            click.echo(f"[compress] consolidated metadata on {merged_path} ({len(names)} array(s): {', '.join(names)})")
        elif utils_cli.drop_consolidated_metadata(merged_path):
            click.echo("[compress] dropped the store's consolidated metadata (--no-consolidate): it would "
                       "now describe this run's fields wrongly.  Readers scan the arrays until the next "
                       "consolidation (dc_toolkit merge_compressed_fields DATASET WHERE_TO_WRITE).")
    utils_cli.write_json(os.path.join(opts.where_to_write, "batch_manifest.json"), {
        "command": "compress", "dataset_file": os.fspath(opts.dataset_file),
        "where_to_write": os.fspath(opts.where_to_write), "merged_store": merged_path,
        "results": results, "any_error": any_error, "env": utils_cli.env_versions(),
    }, "compress")
    if any_error:
        sys.exit(1)


@cli.command("merge_compressed_fields")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("compressed_files_location", type=click.Path(dir_okay=True, file_okay=False, exists=False))
def merge_compressed_fields(dataset_file: str, compressed_files_location: str):
    """Consolidate metadata on {compressed_files_location}/{dataset}.zarr,
    after discarding the unfinished write of an interrupted run (what compress
    does at its end)."""
    utils_cli.require_single_process("merge_compressed_fields")
    merged_path = utils_cli.merged_store_path(compressed_files_location, dataset_file)
    if not Path(merged_path).is_dir():
        raise click.ClickException(f"store not found: {merged_path}.  Did compress run with the same directory?")
    names = utils_cli.consolidate_store(merged_path)
    click.echo(f"[merge] consolidated metadata on {merged_path} ({len(names)} array(s): {', '.join(names)})")


# =============================================================================
# 3. STORE UTILITIES & FORMAT CONVERSION
# =============================================================================

@cli.command("open_zarr_and_inspect")
@click.argument("zarr_path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--head", type=click.IntRange(min=1), default=4, show_default=True,
              help="Elements per dim to preview from each array (0 = metadata only).")
def open_zarr_and_inspect(zarr_path: str, head: int):
    """Print the group tree, per-array metadata (codecs, sharding, ratio) and a
    tiny head slice of a zarr v3 store."""
    utils_cli.require_single_process("open_zarr_and_inspect")
    group, _store = utils.open_zarr_localstore(zarr_path, read_only=True)
    click.echo(group.tree())
    click.echo("-" * 80)
    for name in group.array_keys():
        z = group[name]
        click.echo(f"Array: {name}")
        click.echo(z.info_complete())
        if head > 0:
            slicer = tuple(slice(0, min(head, s)) for s in z.shape)
            click.echo(f"Head slice {slicer}:")
            click.echo(z[slicer])
        click.echo("-" * 80)


@cli.command("from_nc_to_zarr")
@click.argument("nc_path", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.option("--out", "out_zarr", type=click.Path(dir_okay=True, file_okay=False), default=None,
              help="Output .zarr directory (default: input path with .zarr).")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="Remove an existing output directory first.")
@click.option("--consolidated/--no-consolidated", default=True, show_default=True,
              help="Write consolidated metadata.")
@click.option("--preserve-source-chunks/--no-preserve-source-chunks", default=True, show_default=True,
              help="Map each HDF5 chunk 1:1 to a zarr chunk (chunks={}).  Use --no-preserve-source-chunks "
                   "for netCDF-3 or contiguous variables (chunks='auto').")
@click.option("--mask-and-scale/--no-mask-and-scale", default=False, show_default=True,
              help="Apply CF scale_factor/add_offset/_FillValue at read time.  Off keeps packed ints packed "
                   "on disk (the attrs ride along, so readers still decode).")
@click.option("--decode-times/--no-decode-times", default=False, show_default=True,
              help="Apply CF time decoding at read time.  Off keeps the on-disk numeric form.")
@click.option("--threads", type=click.IntRange(min=1), default=None, help="Dask workers (default: visible cores).")
@click.pass_context
def from_nc_to_zarr(ctx, **_):
    """
    Convert a NetCDF file to an UNCOMPRESSED zarr v3 store (no filters, no
    compressors, no sharding; coordinates included; whatever compression the
    netCDF had is undone), for filesystem-level deduplication experiments.
    """
    utils_cli.require_single_process("from_nc_to_zarr")
    utils_cli.nc_to_zarr(utils_cli.opts(ctx))


@cli.command("from_zarr_to_netcdf")
@click.argument("zarr_path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--out", "out_nc", type=click.Path(dir_okay=False), default=None,
              help="Output NetCDF file (default: input path with .nc).")
@click.option("--max-size", default="50GB", callback=utils_cli.size_option_callback, show_default=True,
              help="Refuse to write when the logical output exceeds this size.")
@click.option("--compression", type=click.Choice(["zlib", "none"], case_sensitive=False), default="zlib",
              show_default=True, help="NetCDF variable compression.")
@click.option("--complevel", type=click.IntRange(0, 9), default=4, show_default=True, help="zlib compression level.")
@click.option("--threads", type=click.IntRange(min=1), default=None, help="Dask workers (default: visible cores).")
@click.pass_context
def from_zarr_to_netcdf(ctx, **_):
    """Convert a zarr v3 store to a NetCDF4 file, streamed through dask."""
    utils_cli.require_single_process("from_zarr_to_netcdf")
    utils_cli.zarr_to_netcdf(utils_cli.opts(ctx))


# =============================================================================
# 4. ANALYSIS & PLOTTING
# =============================================================================

@cli.command("perform_clustering")
@click.argument("parquet_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("l_error", type=click.Choice(utils_cli.L_ERRORS))
def perform_clustering(parquet_file: str, l_error: str):
    """
    Elbow and silhouette scores of KMeans (k = 3..9) on compression ratio vs
    the chosen error, over the kept rows of a results_{var}.parquet.

    \b
    Args:
        parquet_file: results_{var}.parquet written by evaluate_combos
        l_error:      "L1", "L2" or "LInf"
    """
    df = utils_cli.load_results(parquet_file)
    if len(df) < 4:
        click.echo(f"[perform_clustering] only {len(df)} finite passing combo(s) in {Path(parquet_file).name}; "
                   f"need >= 4 to cluster.  Nothing to plot.")
        return
    utils_cli.elbow_silhouette_plot(df, l_error)


@cli.command("analyze_clustering")
@click.argument("parquet_file", type=click.Path(exists=True, dir_okay=False))
def analyze_clustering(parquet_file: str):
    """Interactive KMeans scatter plots of L1 / L2 / LInf vs compression ratio
    (opens in the browser) over the kept rows of a results_{var}.parquet."""
    import plotly.io as pio

    df = utils_cli.load_results(parquet_file)
    if len(df) == 0:
        click.echo(f"[analyze_clustering] no finite passing combos in {Path(parquet_file).name}; nothing to plot.")
        return
    pio.renderers.default = "browser"
    utils_cli.clustering_figure(df).show()


@cli.command("plot_compression_errors")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("where_to_write", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.argument("field_to_compress")
@click.option("--pipeline", default=None,
              help="Pipeline to plot: a JSON object with compressor, filter and serializer, or the path "
                   "of a file holding one, incl. a manifest_{field}.json "
                   "(default: the best of manifest_{field}.json in --manifest-dir).")
@click.option("--manifest-dir", default=None,
              help="Directory holding manifest_{field}.json from evaluate_combos (default: WHERE_TO_WRITE).")
@click.pass_context
def plot_compression_errors(ctx, **_):
    """
    Save a 3x3 PDF of compression errors for one (lat, lon) field with one
    pipeline, including a copy shifted by 180 degrees in longitude to reveal
    whether the pipeline respects periodicity.
    """
    opts = utils_cli.opts(ctx)
    field = opts.field_to_compress
    utils_cli.require_single_process("plot_compression_errors")
    os.makedirs(opts.where_to_write, exist_ok=True)
    da = utils.open_dataset(opts.dataset_file, field)[field].squeeze()
    click.echo(f"Squeezed (lat, lon) field_to_compress.nbytes = {utils.hsize(da.nbytes)}")
    if not utils.is_lat_lon(da):
        raise click.ClickException(f"Field {field} must have dimensions (lat, lon); it has {da.dims}.")
    if da.nbytes / 2**30 > 2.5:
        raise click.ClickException(f"Field {field} is too large ({utils.hsize(da.nbytes)}); max 2.5 GiB.")

    combo = utils_cli.plot_pipeline(field, opts.pipeline, opts.manifest_dir or opts.where_to_write)
    utils_cli.validate_pipeline(combo, da, field)
    click.echo(f"pipeline: {utils.pipeline_name(*combo)}")
    da, panels = utils_cli.error_plot_panels(da, field, combo)
    utils_cli.save_error_plot(field, da, panels, os.path.join(opts.where_to_write, f"{field}_compression_errors.pdf"))


# =============================================================================
# 5. UIs & HELP
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))


@cli.command("run_web_ui")
def run_web_ui():
    """Streamlit web UI launched from a local terminal (sweeps run under mpirun, one rank per core)."""
    subprocess.run(["streamlit", "run", os.path.join(_HERE, "compression_analysis_ui_web.py")])


@cli.command("run_web_ui_vcluster")
@click.option("--user_account", type=str, required=True, help="vCluster account (the sweep runs under srun).")
@click.option("--uenv_image", type=str, default="", help="vCluster uenv image name")
@click.option("--uploaded_file", type=str, required=True,
              help="netCDF file on the cluster to analyse (the compute nodes must see it).")
@click.option("--time", type=str, default="00:15:00", help="Allocated time")
@click.option("--nodes", type=str, default="1", help="Number of nodes")
@click.option("--ntasks-per-node", type=str, default="32", help="MPI ranks per node (one per core)")
@click.option("--partition", type=str, default="debug", show_default=True, help="SLURM partition")
def run_web_ui_vcluster(user_account, uenv_image, uploaded_file, time, nodes, ntasks_per_node, partition):
    """The same web UI, launching its sweeps with srun on a vcluster."""
    subprocess.run(["streamlit", "run", os.path.join(_HERE, "compression_analysis_ui_web.py"), "--",
                    "--user_account", user_account, "--uenv_image", uenv_image, "--uploaded_file", uploaded_file,
                    "--time", time, "--nodes", nodes, "--ntasks-per-node", ntasks_per_node,
                    "--partition", partition])


@cli.command("run_local_ui")
def run_local_ui():
    """Desktop (Qt) UI launched from a local terminal."""
    subprocess.run([sys.executable, os.path.join(_HERE, "compression_analysis_ui_local.py")])


@cli.command("help")
@click.pass_context
def help(ctx):
    """Print the help of every command."""
    for command in cli.commands.values():
        if command.name == "help" or command.hidden:
            continue
        click.echo("-" * 80)
        click.echo()
        with click.Context(command, parent=ctx.parent, info_name=command.name) as sub:
            click.echo(command.get_help(ctx=sub))
        click.echo()


if __name__ == "__main__":
    cli()
