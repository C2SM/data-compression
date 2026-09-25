"""The zarr codecs dc_toolkit adds: ZFPY at a chunk's own rank or flattened to 1-D, and EBCC.  A zarr
client reads their arrays through the "zarr.codecs" entry points without MPI or the rest of the package."""
import asyncio
import importlib
import math
import os
import struct

import numcodecs
import numcodecs.zfpy
import numpy as np
from zarr.codecs import numcodecs as zarrcodecs_nc
from zarr.codecs.numcodecs._codecs import _NumcodecsArrayBytesCodec
from zarr.registry import register_codec

os.environ.setdefault("EBCC_LOG_LEVEL", "4")  # the C library logs to stderr; 4 = errors only
try:
    importlib.import_module("ebcc.zarr_filter")  # registers "ebcc_filter" with numcodecs
    from ebcc.filter_wrapper import EBCC_Filter
    EBCC_AVAILABLE = True
except Exception:  # not installed, or a broken install (its C library missing): the other codecs still load
    EBCC_AVAILABLE = False


# ---- ZFPY: two encoders, because neither rank wins --------------------------
# zfp codes 4^d blocks, so the rank a chunk is encoded at decides which correlations it uses:
# the chunk's own rank keeps cross-axis gradients in a block and wins on correlated slower axes;
# 1-D wins when those carry little signal (no bits spent on noise, more redundancy left for the
# compressor).  The sweep carries both, under distinct names: a pipeline's identity is its JSON.


# ZFPYRank encodes each chunk at its own rank, folded only as far as zfp requires (at most 4-D;
# per-axis header budget 2**24 at 2-D, 2**16 at 3-D, 2**12 at 4-D, which DYAMOND's cell axis
# overflows): size-1 axes dropped, then the slowest pair folded until everything fits.  Its
# plain "zfpy" name decodes anywhere.  (Comments, not docstrings: zarr replaces codec docstrings.)
class ZFPYRank(zarrcodecs_nc.ZFPY, codec_name="zfpy"):
    _ZFP_MAX_PER_AXIS = {1: 2**48, 2: 2**24, 3: 2**16, 4: 2**12}

    @classmethod
    def encode_shape(cls, shape) -> tuple:
        dims = tuple(d for d in shape if d > 1) or (1,)
        max_rank = max(cls._ZFP_MAX_PER_AXIS)
        while len(dims) > 1 and (len(dims) > max_rank
                                 or any(d > cls._ZFP_MAX_PER_AXIS[len(dims)] for d in dims)):
            dims = (dims[0] * dims[1],) + dims[2:]     # C-order keeps the fold contiguous
        return dims

    async def _encode_single(self, chunk_data, chunk_spec):
        arr = np.ascontiguousarray(chunk_data.as_ndarray_like())
        out = await asyncio.to_thread(self._codec.encode, arr.reshape(self.encode_shape(arr.shape)))
        return chunk_spec.prototype.buffer.from_bytes(out)


class _ZFPYFlatCodec(numcodecs.zfpy.ZFPY):
    """zfpy under a second numcodecs id: zarr's wrapper resolves codec_name in that registry."""

    codec_id = "zfpy_flat"


numcodecs.register_codec(_ZFPYFlatCodec)


# ZFPYFlat encodes every chunk as 1-D.  Decoding is stock zfp (the stream carries
# its own shape), but a client without dc_toolkit cannot resolve the name.
class ZFPYFlat(ZFPYRank, codec_name="zfpy_flat"):
    @classmethod
    def encode_shape(cls, shape) -> tuple:
        return (max(1, int(np.prod(shape))),)


register_codec("numcodecs.zfpy_flat", ZFPYFlat)   # in-process reads must not depend on the install's entry points


# ---- EBCC (optional): JPEG 2000 base layer + error-bounded residual ----------
# Compresses float32 (lat, lon) frames, each chunk exactly one tile.  NaN/Inf or a tile that
# does not divide the frame make the C library EXIT THE PROCESS, so callers validate first
# (utils.ebcc_tile and ebcc_sweep_entries in the sweep, utils_cli.validate_pipeline in compress).
# Start of both rate-control searches (OpenJPEG rate = base_cr/2): it sets the bisection bracket,
# so it shifts the achieved ratio by ~10 %, and it is part of the arglist, hence the pipeline's identity.
_EBCC_BASE_CR = 2.0
# EBCC_MIN/MAX_INTERNAL_IMAGE_DIM in ebcc_codec.h: the C filter exits outside them.
_EBCC_TILE_MIN, _EBCC_TILE_MAX = 32, 2047


def _f32(bits) -> float:
    """The float32 EBCC packs into a uint32 arglist entry."""
    return struct.unpack("f", struct.pack("I", int(bits)))[0]


class EBCC(_NumcodecsArrayBytesCodec, codec_name="ebcc_filter"):
    """zarr v3 wrapper of ebcc.zarr_filter.EBCCZarrFilter, "numcodecs.ebcc_filter" in zarr.json with
    the integer arglist [height, width, f32bits(base_cr), mode, f32bits(target)]; mode 0 none (no
    target), 1 max_error_target, 2 relative_error_target.  Source only: zarr replaces __doc__."""

    def __init__(self, **codec_config):
        if not EBCC_AVAILABLE:
            raise ImportError("the EBCC serializer needs the ebcc package: pip install -e '.[ebcc]'")
        arglist = list(codec_config.get("arglist") or [])
        try:  # anything else makes the C library exit the process
            mode = int(arglist[3])
            ok = (mode in (0, 1, 2) and len(arglist) == (4 if mode == 0 else 5)
                  and all(_EBCC_TILE_MIN <= int(v) <= _EBCC_TILE_MAX for v in arglist[:2])
                  and all(math.isfinite(_f32(v)) and _f32(v) > 0 for v in arglist[2:3] + arglist[4:5]))
        except (IndexError, TypeError, ValueError, struct.error):
            ok = False
        if not ok:
            raise ValueError(f"EBCC arglist must be [height, width, base_cr bits, mode 0/1/2, target bits (modes 1, 2)] "
                             f"with the tile sides in [{_EBCC_TILE_MIN}, {_EBCC_TILE_MAX}] and positive base_cr and "
                             f"target; got {arglist}")
        super().__init__(**codec_config)

    @classmethod
    def from_params(cls, height: int, width: int, target: float, mode: str = "max_error_target",
                    base_cr: float = _EBCC_BASE_CR):
        opts = EBCC_Filter(base_cr=base_cr, height=height, width=width,
                           residual_opt=(mode, target)).hdf_filter_opts
        return cls(arglist=[int(v) for v in opts])

    @property
    def arglist(self) -> list:
        return [int(v) for v in self.codec_config["arglist"]]

    @property
    def height(self) -> int:
        return self.arglist[0]

    @property
    def width(self) -> int:
        return self.arglist[1]

    def __repr__(self) -> str:
        a = self.arglist
        mode = {0: "none", 1: "max_error_target", 2: "relative_error_target"}.get(a[3], a[3])
        target = f", {mode}={_f32(a[4]):g}" if len(a) > 4 else ""
        return f"EBCC(height={a[0]}, width={a[1]}, base_cr={_f32(a[2]):g}{target})"


register_codec("numcodecs.ebcc_filter", EBCC)
