import json


class Cluster:
    """Cluster config/launch skeleton."""

    def __init__(self, config="sneppx-dist.json"):
        self.config = config

    def init(self):
        data = {"nodes": [{"rank": 0, "gpus": 1}], "backend": "nccl"}
        with open(self.config, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    def status(self):
        try:
            with open(self.config, encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            return {"state": "unknown"}

    def teardown(self):
        return {"state": "stopped"}
