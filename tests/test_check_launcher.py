import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sneppx_dist.cluster import Cluster, detect_backend  # noqa: E402
from sneppx_dist import cli  # noqa: E402


def test_detect_backend_platform():
    assert detect_backend() in {"nccl", "gloo"}


def test_init_default_backend(tmp_path):
    c = Cluster(config=str(tmp_path / "c.json"))
    c.init()
    assert c.status()["backend"] == detect_backend()


def test_check_reachable(tmp_path):
    # bind a local TCP listener on an ephemeral port
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    try:
        c = Cluster(config=str(tmp_path / "c.json"))
        c.init(master_addr="127.0.0.1", master_port=port)
        result = c.check()
        assert result["reachable"] is True
    finally:
        sock.close()


def test_check_unreachable(tmp_path):
    c = Cluster(config=str(tmp_path / "c.json"))
    c.init(master_addr="127.0.0.1", master_port=1)
    result = c.check()
    assert result["reachable"] is False


def test_check_no_config(tmp_path):
    c = Cluster(config=str(tmp_path / "missing.json"))
    try:
        c.check()
        assert False, "should have raised"
    except Exception as exc:
        assert "init" in str(exc)


def test_render_bash_launcher(tmp_path):
    c = Cluster(config=str(tmp_path / "c.json"))
    c.init(world_size=4, backend="gloo", master_port=29511)
    out = c.render_launcher(tmp_path / "train.sh", "train.py", ["--epochs", "5"])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "exec \\\n" in text
    assert "torchrun" in text
    assert "--nproc_per_node=4" in text
    assert "--master_port=29511" in text
    assert "train.py --epochs 5 \"$@\"" in text


def test_render_powershell_launcher(tmp_path):
    c = Cluster(config=str(tmp_path / "c.json"))
    c.init(world_size=2, backend="nccl")
    out = c.render_launcher(tmp_path / "train.ps1", "train.py")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env pwsh")
    assert "& torchrun" in text
    assert "--master_port=29500" in text


def test_cli_launcher(tmp_path, capsys):
    cfg = tmp_path / "cl.json"
    cli.main(["--config", str(cfg), "init", "--world-size", "2", "--backend", "gloo"])
    rc = cli.main(["--config", str(cfg), "launcher", str(tmp_path / "run.sh"), "train.py"])
    assert rc == 0
    assert "launcher written" in capsys.readouterr().out
    assert (tmp_path / "run.sh").exists()