"""A rank's working set for one pipeline, against the model the sweep sizes its sample with
(utils_cli.PER_RANK_WORKING_FACTOR x sample per rank).  Each case runs in a fresh interpreter, since freed
memory an earlier test left resident would hide part of the peak.  Linux only (VmHWM after
/proc/self/clear_refs).  1 MiB chunks on a 256 MiB sample keep production's chunk-to-sample proportion
(16 MiB chunks, 2.8 GiB sample)."""
import os
import subprocess
import sys

import pytest

from dc_toolkit import utils_cli

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not os.path.exists("/proc/self/clear_refs"), reason="needs Linux /proc")]

CASE = r"""
import gc, sys
import numpy as np
from zarr.codecs import numcodecs as nc
from dc_toolkit import utils

def status(key):
    with open("/proc/self/status") as fh:
        return next(int(line.split()[1]) * 1024 for line in fh if line.startswith(key))

kind = sys.argv[1]
shape = (3, 4, 256 * 2**20 // 4 // 12)
rng = np.random.default_rng(0)
if kind.startswith("noise"):
    sample = rng.standard_normal(shape, dtype=np.float32)
else:
    sample = np.broadcast_to(np.sin(np.linspace(0, 60, shape[2], dtype=np.float32)), shape).copy()
cfg = {"noise-plain": (None, None, None), "noise-zstd": (nc.Zstd(level=6), None, None),
       "smooth-bitround": (nc.Zstd(level=6), nc.BitRound(keepbits=7), None)}[kind]
chunks = utils.compute_chunk_shape_for_eval(shape, sample.dtype, target_mib=1, dims=("time", "height", "ncells"))
gc.collect()
with open("/proc/self/clear_refs", "w") as fh:
    fh.write("5")
base = status("VmRSS:")
ratio, _, _ = utils.evaluate_codec_pipeline(sample, ("time", "height", "ncells"), utils.codec_pipeline_kwargs(*cfg),
                                            chunks=chunks)
print("RESULT", ratio, (status("VmHWM:") - base) / sample.nbytes)  # utils prints a timing table at exit
"""


@pytest.mark.parametrize("kind", ["noise-plain", "noise-zstd", "smooth-bitround"])
def test_rank_working_set_within_the_model(kind):
    out = subprocess.run([sys.executable, "-c", CASE, kind], capture_output=True, text=True, check=True).stdout
    ratio, peak = map(float, next(line for line in out.splitlines() if line.startswith("RESULT")).split()[1:])
    print(f"{kind}: ratio {ratio:.2f}, peak above the sample {peak:.2f} x S (1 + 1/ratio = {1 + 1 / ratio:.2f})")
    assert peak <= 1 + 1 / ratio + 0.25, "a rank materialises more than the decoded copy and the encoded bytes"
    if ratio >= 1.2:
        assert peak <= utils_cli.PER_RANK_WORKING_FACTOR, "the sweep's memory model no longer covers one rank"
