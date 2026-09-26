"""dc_toolkit in this process, dying by `signal` (argv[2]: KILL or TERM) while it evaluates a pipeline whose codecs'
repr contains argv[1]; with argv[3] a marker file, only while that file does not exist (it is created first)."""
import os
import signal
import sys

from dc_toolkit import utils
from dc_toolkit.cli import cli

target, how, marker, argv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
evaluate = utils.evaluate_codec_pipeline


def crashing(sample_np, dims, codec_kwargs, *args, **kwargs):
    if target in repr(codec_kwargs) and not (marker != "-" and os.path.exists(marker)):
        if marker != "-":
            open(marker, "w").close()
        os.kill(os.getpid(), getattr(signal, f"SIG{how}"))
    return evaluate(sample_np, dims, codec_kwargs, *args, **kwargs)


utils.evaluate_codec_pipeline = crashing
cli.main(argv, prog_name="dc_toolkit")
