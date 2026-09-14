import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sneppx_dist.cluster import Cluster, ClusterError  # noqa: E402


def test_init_and_status(tmp_path):
    p = tmp_path / "cluster.json"
    c = Cluster(config=str(p))
    c.init(world_size=4, backend="gloo")
    s = c.status()
    assert s["state"] == "created"
    assert s["world_size"] == 4
    assert s["backend"] == "gloo"
    assert p.exists()


def test_start_then_teardown(tmp_path):
    c = Cluster(config=str(tmp_path / "c.json"))
    c.init(world_size=2)
    c.start()
    assert c.status()["state"] == "running"
    c.teardown()
    assert c.status()["state"] == "stopped"


def test_double_start_raises(tmp_path):
    c = Cluster(config=str(tmp_path / "c.json"))
    c.init(world_size=1)
    c.start()
    try:
        c.start()
        assert False, "should have raised"
    except ClusterError:
        pass


def test_launch_command(tmp_path):
    c = Cluster(config=str(tmp_path / "c.json"))
    c.init(world_size=8, master_port=12345)
    cmd = c.launch_command("train.py", ["--epochs", "10"])
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=8" in cmd
    assert "--master_port=12345" in cmd
    assert cmd[-3:] == ["train.py", "--epochs", "10"]


def test_invalid_backend():
    try:
        Cluster().init(backend="xpu")
        assert False, "should have raised"
    except ClusterError:
        pass