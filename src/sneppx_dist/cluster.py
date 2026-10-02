"""Cluster configuration, validation, and torchrun-compatible launch commands.

All state lives in a JSON config file (default ``sneppx-dist.json``). No
runtime dependency on PyTorch - the launcher string is generated so the
user runs it in a torchrun-compatible environment.
"""

import json
import pathlib
import socket
import sys


class ClusterError(Exception):
    """Raised on invalid cluster state."""


_DEFAULT_CONFIG = "sneppx-dist.json"

_BACKENDS = {"nccl", "gloo", "mpi"}

_STATES = {"created", "running", "stopped"}


def detect_backend():
    """Return the recommended distributed backend for this platform.

    NCCL requires Linux + NVIDIA GPUs. Gloo works everywhere but is slower.
    """
    if sys.platform.startswith("linux"):
        return "nccl"
    return "gloo"


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
         backend=None, config=None):
        """Create a fresh cluster config (rejects if already ``running``).

        If *backend* is not supplied, ``detect_backend()`` selects nccl
        on Linux and gloo on Windows/macOS.

        `world_size` = ``gpus_per_node`` × ``nodes`` when ``gpus_per_node`` is set,
        otherwise `world_size` is used directly (or defaults to 1).
        """
        if self._data and self._data.get("state") == "running":
            raise ClusterError("cluster already running; call teardown first")

        backend = (backend or detect_backend()).lower()
        if backend not in _BACKENDS:
            raise ClusterError(f"unknown backend: {backend} (expected one of {_BACKENDS})")

        if gpus_per_node is not None:
            world_size = world_size or gpus_per_node
        else:
            world_size = world_size or 1
        nodes = []
        for i in range(world_size):
            nodes.append({"rank": i, "gpus": (gpus_per_node or 1), "node": f"node-{i}", "address": master_addr})

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

    def reset(self):
        """Reset cluster state to ``created`` (from ``stopped`` or ``running``)."""
        if self._data is None:
            raise ClusterError("no config; run init first")
        self._data["state"] = "created"
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

    # -- connectivity check -------------------------------------------------

    def check(self):
        """Attempt a TCP connect to master_addr:master_port.

        Returns ``{"reachable": bool, "master": ..., "port": ...}`` with
        per-node results aggregated.
        """
        if self._data is None:
            raise ClusterError("no config; run init first")
        try:
            sock = socket.create_connection(
                (self._data["master_addr"], self._data["master_port"]),
                timeout=3,
            )
            sock.close()
            reachable = True
        except OSError:
            reachable = False
        return {
            "reachable": reachable,
            "master": self._data["master_addr"],
            "port": self._data["master_port"],
        }

    # -- launcher rendering -------------------------------------------------

    def render_launcher(self, path, script, script_args=None):
        """Write a runnable shell/powershell launcher to *path*.

        The file is a self-contained script that executes the
        ``torchrun`` command for the saved cluster config + *script*.
        If *path* ends in ``.ps1`` a PowerShell script is generated;
        otherwise a Bash script is written.
        """
        path = pathlib.Path(path)
        cmd = self.launch_command(script, script_args)
        if path.suffix.lower() == ".ps1":
            escaped = []
            for c in cmd:
                if " " in c or ";" in c:
                    escaped.append(f'"{c}"')
                else:
                    escaped.append(c)
            lines = [
                "#!/usr/bin/env pwsh",
                "Set-StrictMode -Version Latest",
                "$ErrorActionPreference='Stop'",
                "",
                "& " + " ".join(escaped) + ' @args',
            ]
        else:
            lines = [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "exec \\" ]
            lines.append("  " + " ".join(cmd) + ' "$@"')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    # -- helpers ------------------------------------------------------------

    def __repr__(self):
        return f"Cluster(path={self._path}, state={self._data and self._data['state']})"