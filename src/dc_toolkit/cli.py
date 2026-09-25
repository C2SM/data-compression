"""dc_toolkit command-line interface: the commands and their options only.
The work happens in utils_cli.py (command helpers) and utils.py (library).

Pipeline
  evaluate_combos   sweep compressor x filter x serializer on a sample
                    -> results_{var}.parquet and manifest_{var}.json (best pipeline)
  compress          persist every swept field (or --vars, or a --pipeline of your
                    own) into {dataset}.zarr and consolidate the store's metadata
  merge_compressed_fields
                    consolidate {dataset}.zarr after compress --no-consolidate runs

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

import click

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
                      "their only filter and get it for any class but 'none' (with a message when the class "
                      "names another filter); a float field the class has no filter for is skipped (an error "
                      "when it is the --field-to-compress)."),
    click.option("--serializer-class", default="all", show_default=True,
                 type=click.Choice(["all", "none", "pcodec", "zfpy", "ebcc"], case_sensitive=False),
                 help="Restrict the serializers to one class; 'all' includes plain bytes, 'none' is plain bytes "
                      "alone, and --with-ebcc adds EBCC to any class.  Fields the class cannot take are "
                      "skipped, as for --filter-class."),
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
_CHUNK_OVERRIDE_OPTIONS = [
    click.option("--inner-chunk-mib", type=click.IntRange(min=1), default=None,
                 help="Target zarr chunk size in MiB (default: the sweep's value, else 16)."),
    click.option("--max-inner-chunk-mib", type=click.IntRange(min=1), default=None,
                 help="Warn above this inner chunk size (MiB) with --no-spatial-split (default: the sweep's, else 256)."),
    click.option("--spatial-split/--no-spatial-split", default=None,
                 help="Split spatial dims when one timestep exceeds --inner-chunk-mib (default: the sweep's, else on)."),
]
_OVERSUBSCRIPTION_OPTION = click.option(
    "--oversubscription-check/--no-oversubscription-check", default=True, show_default=True,
    help="Abort at startup unless " + ", ".join(utils.THREAD_ENV_VARS) + " are all set to 1 "
         "(--no-oversubscription-check only warns).")
_MEMORY_OPTION = click.option(
    "--memory-threshold", type=click.FloatRange(0.05, 0.95), default=0.80, show_default=True,
    help="Max fraction of memory an estimated footprint may use.  evaluate_combos shrinks its sample until "
         "one node's footprint fits this fraction of the node's budget (cgroup limit, else RAM) and aborts "
         "when the field's smallest sample does not fit; compress refuses a write whose peak exceeds this fraction of the available RAM.")
_PERSIST_OPTIONS = _CHUNK_OVERRIDE_OPTIONS + [
    click.option("--shard-mib", type=click.IntRange(min=1), default=512, show_default=True,
                 help="Target shard size in MiB (an integer number of inner chunks).  Sharding is "
                      "skipped when a shard would hold fewer than two chunks."),
    click.option("--threads", type=click.IntRange(min=1), default=None,
                 help="Dask workers for the write (default and maximum: the visible cores).  Peak memory ~ "
                      "threads x (max(source block, shard) + 3 x shard), at most 3 x the field; the memory "
                      "guard refuses what does not fit."),
    _OVERSUBSCRIPTION_OPTION, _MEMORY_OPTION,
]


_VERIFY_OPTIONS = [
    click.option("--verify/--no-verify", default=True, show_default=True,
                 help="Re-read the store after writing and recompute the error norms "
                      "(roughly doubles wall time; skip for trusted re-runs)."),
    click.option("--verify-gate/--no-verify-gate", default=True, show_default=True,
                 help="With --verify, fail a field whose production norms exceed the sweep thresholds "
                      "or physical bounds in manifest_{var}.json, or whose round trip changed a cell's "
                      "finiteness (NaN fill written as data; the gradient gate is sweep-only).  "
                      "--no-verify-gate only warns."),
]


# =============================================================================
# 2. PIPELINE COMMANDS
# =============================================================================

@cli.command("evaluate_combos")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--where-to-write", "where_to_write", required=True,
              type=click.Path(dir_okay=True, file_okay=False, exists=False),
              help="Output directory (config_space_{var}.csv, per-rank CSVs, sweep_state_{var}.json, "
                   "results_{var}.parquet, manifest_{var}.json).  Created if missing.")
@click.option("--field-to-compress", default=None,
              help="Field to sweep (default: every non-empty integer/float32/float64 variable with at least "
                   "one dim, CF bounds excepted).")
@click.option("--eval-data-size-limit", default="5GB", callback=utils_cli.size_option_callback, show_default=True,
              help="Budget of the representative sample the combos are scored on (e.g. 5GB, 512MiB): shrunk "
                   "to fit the node's memory, and raised to the field's smallest sample: 3 time steps (3 indices "
                   "across all time-like dims, ensemble members and forecast steps included) and 3 levels, all "
                   "where there are fewer, or the whole field when it has neither.")
@_OVERSUBSCRIPTION_OPTION
@utils_cli.add_options(_CHUNK_OPTIONS)
@_MEMORY_OPTION
@click.option("--l1-threshold", type=click.FloatRange(min=0.0), required=True,
              callback=utils_cli.finite_option_callback,
              help="Relative L1 error budget (e.g. 0.005 = 0.5%).  The anchor for the other gates.")
@click.option("--l2-threshold", type=click.FloatRange(min=0.0), default=None,
              callback=utils_cli.finite_option_callback,
              help="Relative L2 budget (default: 2 x L1).")
@click.option("--linf-threshold", type=click.FloatRange(min=0.0), default=None,
              callback=utils_cli.finite_option_callback,
              help="Relative Linf (worst cell) budget (default: 10 x L1).")
@click.option("--bias-threshold", type=click.FloatRange(min=0.0), default=None,
              callback=utils_cli.finite_option_callback,
              help="Relative bias budget |mean signed error| / mean|orig| (default: 0.5 x L1).")
@click.option("--q99-threshold", type=click.FloatRange(min=0.0), default=None,
              callback=utils_cli.finite_option_callback,
              help="Relative budget over cells with |value| >= the 99th percentile, taken over the non-zero "
                   "values when the plain one is 0 (default: 2 x L1).  Only with --extremes-sensitive.")
@click.option("--l2-gate/--no-l2-gate", default=True, show_default=True, help="Enable the L2 gate.")
@click.option("--linf-gate/--no-linf-gate", default=True, show_default=True, help="Enable the Linf gate.")
@click.option("--bias-gate/--no-bias-gate", default=True, show_default=True, help="Enable the bias gate.")
@click.option("--extremes-sensitive/--no-extremes-sensitive", default=False, show_default=True,
              help="Enable the q99 extreme-tail gate (precip, gusts, CAPE, radiation peaks).")
@click.option("--phys-min", type=float, default=None,
              callback=utils_cli.finite_option_callback, help="Reject combos whose decoded sample dips below this.")
@click.option("--phys-max", type=float, default=None,
              callback=utils_cli.finite_option_callback, help="Reject combos whose decoded sample exceeds this.")
@click.option("--phys-tolerance", type=click.FloatRange(0.0, 1.0), default=0.0, show_default=True,
              callback=utils_cli.finite_option_callback,
              help="Slack for --phys-min/--phys-max as a fraction of the field's value range: a lossy codec "
                   "rings past a bound the field sits on by a hair. Stored in the manifest as an absolute "
                   "value, so compress's verify gate applies the same slack.")
@click.option("--gradient-gate/--no-gradient-gate", default=False, show_default=True,
              help="Enable the spatial-gradient gate: finite differences along the horizontal dims, or the "
                   "non-leading dims of a field without any (one more pass over the sample per combo; for "
                   "winds, pressure).")
@click.option("--gradient-threshold", type=click.FloatRange(min=0.0), default=0.1, show_default=True,
              callback=utils_cli.finite_option_callback,
              help="Max relative L1 error of the finite-difference field (absolute fraction, not x L1).")
@click.option("--gradient-shortcircuit/--no-gradient-shortcircuit", default=True, show_default=True,
              help="Only compute the gradient for combos that already pass the cheap gates.")
@utils_cli.add_options(_CODEC_SPACE_OPTIONS)
@click.option("--sampling-policy", type=click.Choice(["cascade", "balanced"]), default="cascade",
              show_default=True,
              help="How an over-budget field is thinned (only its time and vertical dims; the others are kept "
                   "whole, and at least 3 time steps and 3 levels are kept, as for --eval-data-size-limit): "
                   "'cascade' keeps a floor of "
                   "levels, then spends the budget on time steps; 'balanced' splits it evenly across the time "
                   "and vertical dims.")
@click.option("--vertical-floor", type=click.IntRange(min=3), default=None,
              help="Vertical levels the cascade policy keeps before it adds time steps (default: "
                   "max(4, ceil(log2(n_levels)))); a budget that cannot hold them with 3 time steps lowers "
                   "it, down to 3.")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip combos already recorded in config_space_{var}_rank*.csv: their metrics are reused "
                   "and the gates re-applied with the current thresholds.  A change to what "
                   "sweep_state_{var}.json records (file, sample, sampling and chunk settings, metric "
                   "definitions, library versions, EBCC's env vars) restarts the field.")
@click.option("--max-evals", type=click.IntRange(min=1), default=None,
              help="Cap the Cartesian product (quick test runs); EBCC combos are always included.")
@click.pass_context
def evaluate_combos(ctx, **_):
    """Sweep compressor x filter x serializer combinations on a representative
    sample of each field, gate them on error thresholds, and record every
    result in results_{var}.parquet and the best pipeline in manifest_{var}.json.

    \b
    Parallelism: MPI ranks split the config space and evaluate one pipeline at
    a time each; the ranks of a node share one copy of the sample.  Launch one
    rank per core:
      srun --nodes=N --ntasks-per-node=32 --cpus-per-task=1 dc_toolkit evaluate_combos ...
      mpirun -n 8 dc_toolkit evaluate_combos ...   (a laptop)
    The combos are evaluated in memory; use `compress` to write the winners.
    """
    utils_cli.sweep_dataset(utils_cli.opts(ctx))


_VERIFY_THRESHOLD_OPTIONS = [
    click.option(f"--{k}-threshold", type=click.FloatRange(min=0.0), default=None,
                 callback=utils_cli.finite_option_callback,
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
                   "the run fails while the store still holds any such array (fields outside --vars included).")
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
              help="Log and go on when a field fails instead of stopping at the first failure.  "
                   "The exit status is 1 either way when any field failed.")
@click.option("--consolidate/--no-consolidate", default=True, show_default=True,
              help="Consolidate the store's metadata at the end so readers open it quickly.  "
                   "--no-consolidate drops any earlier consolidated metadata instead, since this run "
                   "would make it stale; merge_compressed_fields consolidates the store afterwards.")
@click.pass_context
def compress(ctx, **_):
    """Persist fields into {WHERE_TO_WRITE}/{dataset}.zarr, one zarr array per
    field, with the pipeline evaluate_combos found best (manifest_{var}.json,
    else the best kept row of results_{var}.parquet) or the --pipeline you pass.
    A field is written into a staging store beside the real one, gated, and
    only then moved in, so a failed or interrupted write never enters the store.
    batch_manifest.json records every field; the exit status is 1 when any
    field failed.  Single process; dask threads parallelise each write."""
    utils_cli.require_single_process("compress")
    utils_cli.compress_fields(utils_cli.opts(ctx))


@cli.command("merge_compressed_fields")
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument("compressed_files_location", type=click.Path(dir_okay=True, file_okay=False, exists=False))
def merge_compressed_fields(dataset_file: str, compressed_files_location: str):
    """Consolidate the metadata of {COMPRESSED_FILES_LOCATION}/{dataset}.zarr,
    as compress does at its end: the step after compress --no-consolidate runs.
    The unfinished write of an interrupted run is discarded first."""
    utils_cli.require_single_process("merge_compressed_fields")
    utils_cli.consolidate_merged_store(dataset_file, compressed_files_location)


# =============================================================================
# 3. STORE UTILITIES & FORMAT CONVERSION
# =============================================================================

@cli.command("open_zarr_and_inspect")
@click.argument("zarr_path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--head", type=click.IntRange(min=1), default=4, show_default=True,
              help="Elements per dim to preview from each array.")
def open_zarr_and_inspect(zarr_path: str, head: int):
    """Print the group tree, per-array metadata (codecs, sharding, ratio) and a
    tiny head slice of a zarr v3 store."""
    utils_cli.require_single_process("open_zarr_and_inspect")
    utils_cli.inspect_store(zarr_path, head)


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
    """Convert a NetCDF file to an UNCOMPRESSED zarr v3 store (no filters, no
    compressors, no sharding; coordinates included; whatever compression the
    netCDF had is undone), for filesystem-level deduplication experiments."""
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
@click.option("--threads", type=click.IntRange(min=1), default=None,
              help="Dask workers (default and maximum: the visible cores).")
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
    """Plot the elbow and silhouette scores of KMeans (k = 3..9, at most rows - 1) on compression ratio vs
    the chosen error, over the kept rows of a results_{var}.parquet from evaluate_combos."""
    utils_cli.perform_clustering(parquet_file, l_error)


@cli.command("analyze_clustering")
@click.argument("parquet_file", type=click.Path(exists=True, dir_okay=False))
def analyze_clustering(parquet_file: str):
    """Interactive KMeans scatter plots of L1 / L2 / LInf vs compression ratio
    (opens in the browser) over the kept rows of a results_{var}.parquet."""
    utils_cli.analyze_clustering(parquet_file)


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
    """Save WHERE_TO_WRITE/{field}_compression_errors.pdf: a 3x3 grid of the
    compression errors of one (lat, lon) field with one pipeline, including a
    copy shifted by 180 degrees in longitude to reveal whether the pipeline
    respects periodicity."""
    utils_cli.require_single_process("plot_compression_errors")
    utils_cli.plot_compression_errors(utils_cli.opts(ctx))


# =============================================================================
# 5. UIs & HELP
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))


@cli.command("run_web_ui")
def run_web_ui():
    """Streamlit web UI launched from a local terminal (sweeps run under mpirun or mpiexec when one is on the
    PATH, one rank per physical core; else as one process)."""
    subprocess.run(["streamlit", "run", os.path.join(_HERE, "compression_analysis_ui_web.py")])


@cli.command("run_web_ui_vcluster")
@click.option("--user_account", type=str, required=True, help="vCluster account (the commands run under srun).")
@click.option("--uenv_image", type=str, default="",
              help="uenv image the commands run in, with its default view (default: srun keeps the calling "
                   "session's uenv, if any).")
@click.option("--uploaded_file", type=str, required=True,
              help="netCDF file on the cluster to analyse (the compute nodes must see it).")
@click.option("--time", type=str, default="00:15:00", help="Time limit of each srun (default: 00:15:00).")
@click.option("--nodes", type=str, default="1", help="Nodes of a sweep (default: 1).")
@click.option("--ntasks-per-node", type=str, default="32",
              help="MPI ranks per node of a sweep, one per core (default: 32).")
@click.option("--partition", type=str, default="debug", show_default=True, help="SLURM partition")
def run_web_ui_vcluster(user_account, uenv_image, uploaded_file, time, nodes, ntasks_per_node, partition):
    """The same web UI, launching its commands with srun on a vcluster (compress as one task)."""
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
