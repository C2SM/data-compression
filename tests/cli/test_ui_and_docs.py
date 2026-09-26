"""The web UI renders and builds valid commands; every flag the docs name exists."""
import pathlib
import re
import sys

import pytest

from conftest import REPO


def _options(command):
    return {o for p in command.params for o in (*p.opts, *p.secondary_opts)}


def test_ui_commands_use_existing_flags():
    from dc_toolkit import utils_cli
    from dc_toolkit.cli import cli
    sweep = utils_cli.ui_sweep_command(["mpirun", "-n", "2"], "f.nc", "out", "t",
                                       {"compressor": "all", "filter": "all", "serializer": "all"}, True, False, 0.005)
    comp = utils_cli.ui_compress_command([], "f.nc", "out", "t", {"compressor": None, "filter": None, "serializer": None})
    for argv, name in ((sweep[sweep.index("dc_toolkit") + 1:], "evaluate_combos"), (comp[1:], "compress")):
        assert argv[0] == name
        assert {a for a in argv if a.startswith("--")} <= _options(cli.commands[name])


def test_web_ui_renders(monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    from dc_toolkit import cli as cli_module
    monkeypatch.setattr(sys, "argv", ["streamlit"])
    at = AppTest.from_file(str(pathlib.Path(cli_module.__file__).with_name("compression_analysis_ui_web.py")),
                           default_timeout=60).run()
    assert not at.exception and at.title[0].value.startswith("Evaluate compressors")


DOCS = ["README.md", "docs/intro.md", "docs/PARALLELIZATION.md", "docs/SAMPLING.md", "santis.run"]
NOT_OURS = {"--nodes", "--ntasks-per-node", "--cpus-per-task", "--account", "--unbuffered", "--shm-size", "--ntasks",
            "--entrypoint", "--allow-run-as-root", "--uenv", "--view", "--partition", "--time", "--mem",
            "--only-binary", "--no-binary", "--quiet", "--output", "--error", "--install", "--distribution"}


@pytest.mark.parametrize("doc", DOCS)
def test_every_documented_flag_exists(doc):
    from dc_toolkit.cli import cli
    ours = set().union(*(_options(c) for c in cli.commands.values())) | {"--help"}
    text = (REPO / doc).read_text()
    unknown = sorted(f for f in set(re.findall(r"(?<![\w-])--[a-z][a-z0-9_-]*", text)) if f not in ours | NOT_OURS)
    assert not unknown, f"{doc} names flags no command has: {unknown}"


@pytest.mark.parametrize("doc", DOCS[:4])
def test_every_documented_log_tag_exists(doc):
    """A [tag] a document quotes from the output must still be printed somewhere (a renamed tag breaks grep)."""
    code = "".join(p.read_text() for p in (REPO / "src" / "dc_toolkit").glob("*.py"))
    code += (REPO / "santis.run").read_text() + (REPO / "install_dc_toolkit.sh").read_text()
    tags = set(re.findall(r"`[^`\n]*?(?<!\.)(\[[a-z][a-z0-9.-]*\])", (REPO / doc).read_text()))  # not ".[extra]"
    tags |= set(re.findall(r"^(\[[a-z][a-z0-9.-]*\])(?!\()", (REPO / doc).read_text(), re.M))  # not a link
    assert not sorted(t for t in tags if t not in code), f"{doc} quotes log tags the code no longer prints"
