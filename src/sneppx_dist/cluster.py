"""Cluster configuration, validation, and torchrun-compatible launch commands.

All state lives in a JSON config file (default ``sneppx-dist.json``). No
runtime dependency on PyTorch - the launcher string is generated so the
user runs it in a torchrun-compatible environment.
"""

import json
import pathlib


class ClusterError(Exception):
    """Raised on invalid cluster state."""


_DEFAULT_CONFIG = "sneppx-dist.json"

_BACKENDS = {"nccl", "gloo", "mpi"}

_STATES = {"created", "running", "stopped"}


class Cluster:
    """Persistent cluster config with validation and torchrun launch helpers."""

    def __init__(self, config=_DEFAULT_CONFIG):
        self._path = pathlib.Path(config)
        self._data = None
        if self._path.exists():
            self._load()

    # -- persistence -------------------------------------------------------

    def _load(self):
        self._data = json.loads(self._path.read_text(encoding="utf-8"))
        return self

    def _save(self):
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        return self

    # -- init --------------------------------------------------------------

    def init(self, *, world_size=None, gpus_per_node=None,
             master_addr="127.0.0.1", master_port=29500,
             backend="nccl", config=None):
        """Create a fresh cluster config (rejects if already ``running``)."""
        if self._data and self._data.get("state") == "running":
            raise ClusterError("cluster already running; call teardown first")

        backend = backend.lower()
        if backend not in _BACKENDS:
            raise ClusterError(f"unknown backend: {backend} (expected one of {_BACKENDS})")

        world_size = world_size or gpus_per_node or 1
        nodes = []
        for i in range(world_size):
            nodes.append({"rank": i, "gpus": 1, "node": f"node-{i}", "address": master_addr})

        self._data = {
            "format": "sneppx-dist-cluster",
            "state": "created",
            "backend": backend,
            "world_size": world_size,
            "master_addr": master_addr,
            "master_port": int(master_port),
            "nodes": nodes,
        }
        if config:
            self._path = pathlib.Path(config)
        self._save()
        return self.status()

    # -- status ------------------------------------------------------------

    def status(self):
        if self._data is None:
            return {"state": "unknown"}
        return dict(self._data)

    # -- start / stop transitions ------------------------------------------

    def start(self):
        if self._data is None:
            raise ClusterError("no config; run init first")
        if self._data["state"] == "running":
            raise ClusterError("already running")
        self._data["state"] = "running"
        self._save()
        return self.status()

    def teardown(self):
        if self._data is None:
            return {"state": "unknown"}
        self._data["state"] = "stopped"
        self._save()
        return self.status()

    # -- torchrun command generation ----------------------------------------

    def launch_command(self, script, script_args=None):
        """Return a ``torchrun``-compatible CLI command list.

        Doesn't execute anything - purely declarative so the user can pipe
        it to a shell or print it.
        """
        if self._data is None:
            raise ClusterError("no config; run init first")
        cmd = [
            "torchrun",
            f"--nproc_per_node={self._data['world_size']}",
            f"--master_addr={self._data['master_addr']}",
            f"--master_port={self._data['master_port']}",
            script,
        ]
        if script_args:
            cmd.extend(script_args)
        return cmd

    # -- helpers ------------------------------------------------------------

    def __repr__(self):
        return f"Cluster(path={self._path}, state={self._data and self._data['state']})"