"""
Validate what santis_test.run produced:  python check_outputs.py RESULTS_BASE

Walks RESULTS_BASE/steps.tsv, checks the outputs of every step against the
invariants listed in CLAUDE.md, writes RESULTS_BASE/test_report.md and exits 1
when any check failed.  It does not import dc_toolkit: gate verdicts, best
pipelines and stored codecs are recomputed here from the recorded files.
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

COLUMNS = [
    "name", "compressor", "filter", "serializer", "pipeline",
    "ratio", "l1_rel", "l2_rel", "linf_rel", "bias_rel", "q99_rel", "grad_rel",
    "decoded_min", "decoded_max", "n_corrupt", "eucd",
    "pass_l1", "pass_l2", "pass_linf", "pass_bias", "pass_q99", "pass_bounds", "pass_grad", "pass_finite",
    "keep",
]
METRICS = ["ratio", "l1_rel", "l2_rel", "linf_rel", "bias_rel", "q99_rel", "grad_rel",
           "decoded_min", "decoded_max", "n_corrupt", "eucd"]
CHEAP = (("l1_rel", "l1"), ("l2_rel", "l2"), ("linf_rel", "linf"), ("bias_rel", "bias"))
PRODUCTION = (("l1", "Relative_Error_L1"), ("l2", "Relative_Error_L2"), ("linf", "Relative_Error_Linf"),
              ("bias", "Bias_Rel"), ("q99", "Q99_Rel"))
RESUME_RE = re.compile(r"\[resume\] (\d+) of (\d+) combo\(s\) of '[^']*' are already recorded")
EXCERPT_RE = re.compile(r"^\[(sample|memcheck|topology|memory|sweep|resume|ebcc|warning|chunks|cr-drift|verify-gate|"
                        r"compress|persist|combo-filter|max-evals|gates|var|oversubscription|store)\]|"
                        r"Traceback|ERROR|FATAL|REFUSING|\bOOM\b|oom-kill|Killed|srun: error", re.IGNORECASE)


class Report:
    def __init__(self):
        self.fails, self.warns, self.fields, self.excerpts = [], [], [], []

    def check(self, ok, where, message):
        if not ok:
            self.fails.append(f"{where}: {message}")
        return bool(ok)

    def warn(self, where, message):
        self.warns.append(f"{where}: {message}")


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def truthy(series):
    return series.astype(str).str.strip().str.lower().isin(("true", "1"))


def num(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def thresholds(manifest):
    return {k: (math.inf if v is None else float(v)) for k, v in (manifest.get("effective_thresholds") or {}).items()}


def rank_files(d, prefix, var):
    pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(var)}_rank\d+\.csv$")
    return sorted(p for p in Path(d).iterdir() if pattern.match(p.name))


def cheap_pass(df, thr):
    ok = pd.Series(True, index=df.index)
    for column, key in CHEAP:
        if math.isfinite(thr.get(key, math.inf)):
            ok &= ~(num(df, column) > thr[key])
    return ok


def expected_keep(df, manifest):
    """The keep verdict every row should carry, from its metrics and the manifest."""
    thr = thresholds(manifest)
    ok = cheap_pass(df, thr)
    if math.isfinite(thr.get("q99", math.inf)):
        ok &= ~(num(df, "q99_rel") > thr["q99"])
    ok &= num(df, "n_corrupt").fillna(0) == 0
    dmin, dmax = num(df, "decoded_min"), num(df, "decoded_max")
    slack = float(manifest.get("phys_slack") or 0.0)
    if manifest.get("phys_min") is not None:
        ok &= ~(np.isfinite(dmin) & (dmin < manifest["phys_min"] - slack))
    if manifest.get("phys_max") is not None:
        ok &= ~(np.isfinite(dmax) & (dmax > manifest["phys_max"] + slack))
    if manifest["args"].get("gradient_gate"):
        ok &= ~(num(df, "grad_rel") > float(manifest["gradient_threshold"]))
    return ok


def best_row(df):
    kept = df[truthy(df["keep"])]
    return None if kept.empty else kept.sort_values(["ratio", "l1_rel", "pipeline"], ascending=[False, True, True]).iloc[0]


def resume_count(log_path):
    """(reused, planned) from the [resume] line of a sweep log; (0, None) without one."""
    try:
        m = RESUME_RE.search(Path(log_path).read_text(errors="replace"))
    except OSError:
        return 0, None
    return (int(m.group(1)), int(m.group(2))) if m else (0, None)


def same_rows(rep, where, df, ref, columns):
    """Every pipeline of `ref` is in `df` with identical values in `columns`."""
    if not rep.check(set(df["pipeline"]) == set(ref["pipeline"]), where,
                     f"{len(set(df['pipeline']) ^ set(ref['pipeline']))} pipeline(s) differ from the reference run"):
        return
    merged = ref.merge(df, on="pipeline", suffixes=("_ref", ""))
    for column in columns:
        a, b = merged[f"{column}_ref"], merged[column]
        if column == "keep":
            bad = int((truthy(a) != truthy(b)).sum())
        else:
            a, b = pd.to_numeric(a, errors="coerce").to_numpy(float), pd.to_numeric(b, errors="coerce").to_numpy(float)
            bad = int((~np.isclose(a, b, rtol=1e-12, atol=0.0, equal_nan=True)).sum())
        rep.check(bad == 0, where, f"{bad} row(s) differ from the reference run in '{column}'")


# -----------------------------------------------------------------------------
# Sweep outputs
# -----------------------------------------------------------------------------

def check_sweep(rep, d, var, where):
    """Invariants of one evaluate_combos output directory; returns a summary dict or None."""
    d = Path(d)
    manifest = load_json(d / f"manifest_{var}.json")
    parquet = d / f"results_{var}.parquet"
    if not rep.check(manifest is not None and parquet.is_file(), where, "manifest or results parquet missing"):
        return None
    df = pd.read_parquet(parquet)
    plan = pd.read_csv(d / f"config_space_{var}.csv")
    state = load_json(d / f"sweep_state_{var}.json")
    n, n_failed, n_passed = manifest["num_combos"], manifest["num_failed_total"], manifest["num_passed"]

    rep.check(state is not None, where, f"sweep_state_{var}.json missing")
    rep.check(list(df.columns) == COLUMNS, where, f"parquet columns differ: {list(df.columns)}")
    rep.check(df["pipeline"].is_unique, where, "duplicate pipelines in the parquet")
    rep.check(len(plan) == n and plan["pipeline"].is_unique, where,
              f"config_space_{var}.csv has {len(plan)} rows (unique: {plan['pipeline'].is_unique}), num_combos={n}")
    rep.check(set(df["pipeline"]) <= set(plan["pipeline"]), where, "parquet holds pipelines outside the planned space")
    rep.check(len(df) + n_failed == n, where, f"{len(df)} result rows + {n_failed} failures != {n} planned combos")
    failures = pd.concat([pd.read_csv(p) for p in rank_files(d, "failures", var)] or [pd.DataFrame()])
    rep.check(len(failures) == n_failed, where, f"failures files hold {len(failures)} rows, manifest says {n_failed}")
    if len(failures):
        rep.warn(where, f"{len(failures)} combo(s) raised, e.g. {failures.iloc[0]['name']}: {str(failures.iloc[0]['error'])[:200]}")

    keep = truthy(df["keep"])
    rep.check(int(keep.sum()) == n_passed, where, f"{int(keep.sum())} kept rows, manifest num_passed={n_passed}")
    rep.check(int(manifest["num_filtered"]) == n - n_passed - n_failed, where, "num_filtered != combos - passed - failed")
    passes = pd.concat([truthy(df[c]) for c in COLUMNS if c.startswith("pass_")], axis=1).all(axis=1)
    rep.check(bool((passes == keep).all()), where, "keep is not the AND of the pass_* columns")
    disagree = int((expected_keep(df, manifest) != keep).sum())
    rep.check(disagree == 0, where, f"{disagree} row(s) whose keep verdict disagrees with their metrics and the manifest thresholds")

    top, best = best_row(df), manifest.get("best")
    if top is None:
        rep.check(best is None, where, "no kept rows, yet the manifest names a best pipeline")
    else:
        rep.check(best is not None and best["pipeline"] == json.loads(top["pipeline"]) and best["name"] == top["name"]
                  and math.isclose(best["ratio"], float(top["ratio"]), rel_tol=1e-12), where,
                  "manifest best is not the top kept row (ratio desc, L1 asc, pipeline)")

    args, thr = manifest["args"], thresholds(manifest)
    q99_abs = manifest.get("q99_abs")
    if args.get("extremes_sensitive") and q99_abs is not None and math.isfinite(q99_abs):
        rep.check(bool(df["q99_rel"].notna().all()), where, "q99 gate on, but some rows lack q99_rel")
    if args.get("gradient_gate"):
        needs = cheap_pass(df, thr)
        rep.check(bool(df.loc[needs, "grad_rel"].notna().all()), where,
                  f"{int(df.loc[needs, 'grad_rel'].isna().sum())} row(s) pass the cheap gates but lack grad_rel")
        extra = int(df.loc[~needs, "grad_rel"].notna().sum())
        if extra:
            rep.warn(where, f"{extra} row(s) failing a cheap gate carry grad_rel (expected none with --gradient-shortcircuit)")
    itemsize = np.dtype((state or {}).get("dtype", "float32")).itemsize
    compressors = [json.loads(p)["compressor"] for p in plan["pipeline"]]
    blosc = [c for c in compressors if c and c["name"] == "numcodecs.blosc"]
    rep.check(all(c["configuration"].get("typesize") == itemsize for c in blosc), where,
              f"a Blosc compressor lacks typesize={itemsize}")
    n_ebcc = int(plan["serializer"].str.startswith("EBCC").sum())
    if args.get("with_ebcc"):
        rep.check(n_ebcc == 7, where, f"--with-ebcc planned {n_ebcc} EBCC combos, expected 7")
    corrupt = int((num(df, "n_corrupt").fillna(0) > 0).sum())
    if corrupt:
        rep.warn(where, f"{corrupt} row(s) decoded finite cells to NaN/Inf (rejected by the finite gate)")
    return {"manifest": manifest, "df": df, "n": n, "passed": n_passed, "failed": n_failed, "n_ebcc": n_ebcc,
            "kept_ebcc": int((keep & df["serializer"].str.startswith("EBCC")).sum()),
            "best": best, "seconds": manifest.get("var_sweep_seconds")}


# -----------------------------------------------------------------------------
# Compress outputs
# -----------------------------------------------------------------------------

def stored_codecs(array_meta):
    codecs = array_meta["codecs"]
    if codecs and codecs[0]["name"] == "sharding_indexed":
        return codecs[0]["configuration"]["codecs"], codecs[0]["configuration"]["chunk_shape"]
    return codecs, None


def check_compress(rep, d, var, source, where, pipeline, thr, phys=(None, None, 0.0), batch_name="batch_manifest.json"):
    """Invariants of one compress run into `d`; returns a summary dict or None."""
    d = Path(d)
    batch = load_json(d / batch_name)
    entry = ((batch or {}).get("results") or {}).get(var)
    if not rep.check(entry is not None, where, f"{batch_name} lacks the field"):
        return None
    rep.check(entry.get("status") == "ok", where, f"status={entry.get('status')} {entry.get('error', '')}")
    rep.check(not batch.get("any_error"), where, "batch_manifest.json any_error=true")
    if entry.get("status") != "ok":
        return None
    rep.check(entry.get("pipeline") == pipeline, where, "batch_manifest.json pipeline is not the expected pipeline")
    rep.check(entry.get("verify_gate") == "pass", where, f"verify_gate={entry.get('verify_gate')}")
    errors = entry.get("errors") or {}
    for key, metric in PRODUCTION:
        limit, value = thr.get(key, math.inf), errors.get(metric)
        if value is not None and math.isfinite(limit):
            rep.check(value <= limit, where, f"production {metric}={value:.3e} exceeds {key}={limit:.3e}")
    rep.check(errors.get("N_Corrupt", 0) == 0, where, f"N_Corrupt={errors.get('N_Corrupt')}")
    slack = float(phys[2] if len(phys) > 2 and phys[2] else 0.0)
    if phys[0] is not None and errors.get("Decoded_Min") is not None:
        rep.check(errors["Decoded_Min"] >= phys[0] - slack, where, f"Decoded_Min={errors['Decoded_Min']} < phys_min={phys[0]} - slack {slack}")
    if phys[1] is not None and errors.get("Decoded_Max") is not None:
        rep.check(errors["Decoded_Max"] <= phys[1] + slack, where, f"Decoded_Max={errors['Decoded_Max']} > phys_max={phys[1]} + slack {slack}")
    drift = entry.get("cr_drift")
    if drift is not None and abs(drift) > 0.25:
        rep.warn(where, f"compression ratio drift {drift:+.1%} (achieved {entry['ratio']:.2f} vs sweep {entry['predicted_ratio']:.2f})")

    store = d / f"{Path(source).stem}.zarr"
    if not rep.check(store.is_dir(), where, f"store {store.name} missing"):
        return None
    rep.check(not Path(f"{store}.__staging__").exists(), where, "a staging store was left behind")
    array_meta, root = load_json(store / var / "zarr.json"), load_json(store / "zarr.json")
    consolidated = ((root or {}).get("consolidated_metadata") or {}).get("metadata") or {}
    rep.check(consolidated.get(var) is not None and consolidated[var].get("codecs") == (array_meta or {}).get("codecs"),
              where, "consolidated metadata does not describe the array as written")
    inner, shard_chunks = stored_codecs(array_meta)
    expected = ([pipeline["filter"]] if pipeline["filter"] else []) + [pipeline["serializer"] or {"name": "bytes"}] \
        + ([pipeline["compressor"]] if pipeline["compressor"] else [])
    same = len(inner) == len(expected) and all(
        s["name"] == e["name"] and (e["name"] == "bytes" or s.get("configuration") == e.get("configuration"))
        for s, e in zip(inner, expected))
    rep.check(same, where, f"stored codecs {json.dumps(inner)[:400]} are not the pipeline {json.dumps(expected)[:400]}")

    src = xr.open_dataset(source, chunks={})[var]
    out = xr.open_zarr(store, consolidated=True)[var]
    rep.check(out.shape == src.shape and out.dtype == src.dtype and out.dims == src.dims, where,
              f"store {out.dims}{out.shape}{out.dtype} != source {src.dims}{src.shape}{src.dtype}")
    slab = {src.dims[0]: 0} if src.ndim > 1 else {src.dims[0]: slice(0, 1024)}
    a, b = src.isel(slab).values.astype(np.float64), out.isel(slab).values.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    rep.check(int((np.isfinite(a) & ~np.isfinite(b)).sum()) == 0, where, "the first slab read back has new NaN/Inf")
    denom = np.abs(a[finite]).sum()
    slab_l1 = float(np.abs(b[finite] - a[finite]).sum() / denom) if denom else 0.0
    store_bytes = sum(p.stat().st_size for p in store.rglob("*") if p.is_file())
    return {"ratio": entry["ratio"], "predicted": entry.get("predicted_ratio"), "drift": drift,
            "verify": entry.get("verify_gate"), "l1": errors.get("Relative_Error_L1"), "slab_l1": slab_l1,
            "store_mib": store_bytes / 2**20, "source_mib": Path(source).stat().st_size / 2**20,
            "chunks": entry.get("inner_chunks"), "shards": entry.get("shards"), "seconds": entry.get("seconds")}


# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------

def excerpt(rep, step, log_path, failed):
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return
    picked = [line for line in lines if EXCERPT_RE.search(line)][:40]
    if failed:
        picked += ["...", *lines[-15:]]
    if picked:
        rep.excerpts.append((f"{step} {log_path}", picked))


def main(base):
    base = Path(base).resolve()
    rep = Report()
    steps = pd.read_csv(base / "steps.tsv", sep="\t", dtype=str, keep_default_na=False)
    sweeps = {}
    for s in steps.itertuples(index=False):
        d = Path(s.dir).resolve()
        where = f"{s.step} {s.key} ({d.relative_to(base) if d.is_relative_to(base) else d})"
        if s.step == "check":
            continue
        if s.rc == "skipped-walltime":
            rep.warn(where, "not run: not enough walltime left")
            continue
        if not rep.check(s.rc.lstrip("-").isdigit(), where, f"not run: {s.rc}"):
            continue
        rc = int(s.rc)
        log_path = d / f"{s.step}.log"
        excerpt(rep, s.step, log_path, rc != 0 and s.step != "kill_run")
        try:
            if s.step == "kill_run":
                if rc == 0:
                    rep.warn(where, "the sweep finished before the kill; the kill/resume check is inconclusive")
                continue
            if not rep.check(rc == 0, where, f"exit status {rc} (see {log_path})"):
                continue
            if s.step in ("preflight", "nodes", "cli_help"):
                continue
            ref_dir = base / "fields" / s.key
            if s.step in ("sweep", "resume_identical", "resume_gradient", "kill_resume"):
                info = check_sweep(rep, d, s.var, where)
                if info is None:
                    continue
                sweeps[str(d)] = info
                rep.fields.append({"where": str(d.relative_to(base)), "sweep": info, "compress": None, "step": s.step})
                ref = sweeps.get(str(ref_dir))
                reused, planned = resume_count(log_path)
                if s.step == "resume_identical" and ref:
                    rep.check(reused == len(ref["df"]) and planned == ref["n"], where,
                              f"[resume] reused {reused} of {planned}, expected {len(ref['df'])} of {ref['n']}")
                    same_rows(rep, where, info["df"], ref["df"], METRICS + ["keep"])
                    rep.check(info["best"] == ref["best"], where, "best pipeline differs from the original sweep")
                elif s.step == "resume_gradient" and ref:
                    thr = thresholds(ref["manifest"])
                    expected = int((ref["df"]["grad_rel"].notna() | ~cheap_pass(ref["df"], thr)).sum())
                    rep.check(reused == expected and planned == ref["n"], where,
                              f"[resume] reused {reused} of {planned}, expected {expected} of {ref['n']} "
                              f"(rows failing a cheap gate need no gradient)")
                    same_rows(rep, where, info["df"], ref["df"], [m for m in METRICS if m != "grad_rel"])
                elif s.step == "kill_resume" and ref:
                    if not 0 < reused < ref["n"]:
                        rep.warn(where, f"[resume] reused {reused} of {planned}: the kill did not land mid-sweep, inconclusive")
                    same_rows(rep, where, info["df"], ref["df"], METRICS + ["keep"])
                    rep.check(info["best"] == ref["best"], where, "best pipeline differs from the uninterrupted sweep")
            elif s.step == "compress":
                info = sweeps.get(str(d))
                if not rep.check(info is not None and info["best"] is not None, where, "no sweep best to compress"):
                    continue
                man = info["manifest"]
                first = d / "batch_manifest.first.json"  # compress_rerun overwrites batch_manifest.json
                out = check_compress(rep, d, s.var, s.input, where, info["best"]["pipeline"], thresholds(man),
                                     (man.get("phys_min"), man.get("phys_max"), man.get("phys_slack") or 0.0),
                                     first.name if first.is_file() else "batch_manifest.json")
                for field in rep.fields:
                    if field["where"] == str(d.relative_to(base)) and field["step"] == "sweep":
                        field["compress"] = out
            elif s.step == "compress_rerun":
                entry = ((load_json(d / "batch_manifest.json") or {}).get("results") or {}).get(s.var) or {}
                rep.check(entry.get("status") == "skipped-existing", where,
                          f"re-running compress gave status={entry.get('status')}, expected skipped-existing")
            elif s.step == "pipeline_compress":
                pipeline = load_json(d / "pipeline.json")
                out = check_compress(rep, d, s.var, s.input, where, pipeline, {"l1": float(s.l1)})
                entry = ((load_json(d / "batch_manifest.json") or {}).get("results") or {}).get(s.var) or {}
                rep.check(entry.get("source") == "--pipeline", where, f"source={entry.get('source')}")
                rep.fields.append({"where": str(d.relative_to(base)), "sweep": None, "compress": out, "step": s.step,
                                   "name": entry.get("name")})
        except Exception as e:  # a crash in one check must not hide the others
            rep.fails.append(f"{where}: checker raised {e!r}")
    write_report(rep, base, steps)
    print((base / "test_report.md").read_text())
    return 1 if rep.fails else 0


def fmt(x, spec=".3g"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return format(int(x), "d") if spec == "d" else format(x, spec)


def write_report(rep, base, steps):
    verdict = "PASS" if not rep.fails else "FAIL"
    lines = [f"# dc_toolkit Santis test report: {verdict}", "",
             f"`{base}`, checked {time.strftime('%Y-%m-%d %H:%M:%S')}: "
             f"{len(rep.fails)} failure(s), {len(rep.warns)} warning(s).", "",
             "## Steps", "", "| step | key | exit | seconds |", "|---|---|---|---|"]
    lines += [f"| {s.step} | {s.key} | {s.rc} | {s.seconds} |" for s in steps.itertuples(index=False)]
    lines += ["", "## Fields", "",
              "| output | combos | kept | failed | EBCC kept | best pipeline | sweep ratio | L1 | sweep s "
              "| store ratio | drift | verify | slab L1 | store MiB | nc MiB |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in rep.fields:
        sw, co = f["sweep"] or {}, f["compress"] or {}
        best = sw.get("best") or {}
        name = best.get("name") or f.get("name") or "-"
        lines.append(f"| {f['where']} | {fmt(sw.get('n'), 'd')} | {fmt(sw.get('passed'), 'd')} | {fmt(sw.get('failed'), 'd')} "
                     f"| {fmt(sw.get('kept_ebcc'), 'd') if sw.get('n_ebcc') else '-'} | `{name}` | {fmt(best.get('ratio'), '.2f')} "
                     f"| {fmt(best.get('l1_rel'))} | {fmt(sw.get('seconds'), '.0f')} | {fmt(co.get('ratio'), '.2f')} "
                     f"| {fmt(co.get('drift'), '+.1%')} | {co.get('verify', '-')} | {fmt(co.get('slab_l1'))} "
                     f"| {fmt(co.get('store_mib'), '.1f')} | {fmt(co.get('source_mib'), '.1f')} |")
    lines += ["", "## Failures", ""] + ([f"- {m}" for m in rep.fails] or ["none"])
    lines += ["", "## Warnings", ""] + ([f"- {m}" for m in rep.warns] or ["none"])
    lines += ["", "## Log excerpts", ""]
    for title, picked in rep.excerpts:
        lines += [f"<details><summary>{title}</summary>", "", "```", *picked, "```", "", "</details>", ""]
    (base / "test_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
