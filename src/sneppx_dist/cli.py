import argparse
import json
import sys

from sneppx_dist.cluster import Cluster, ClusterError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sneppx-dist", description="distributed training CLI")
    parser.add_argument("--version", action="version", version="sneppx-dist 0.1.0")
    parser.add_argument("--config", default="sneppx-dist.json", help="cluster config path")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize a training cluster config")
    init.add_argument("--world-size", type=int, default=1, help="number of processes / GPUs")
    init.add_argument("--backend", default="nccl", choices=["nccl", "gloo", "mpi"])
    init.add_argument("--master-addr", default="127.0.0.1")
    init.add_argument("--master-port", type=int, default=29500)

    sub.add_parser("status", help="show cluster status")

    launch = sub.add_parser("launch", help="print a torchrun launch command")
    launch.add_argument("script", help="training script to run")
    launch.add_argument("script_args", nargs="*", help="args passed through to the script")

    sub.add_parser("start", help="transition cluster state to running")
    sub.add_parser("teardown", help="stop the cluster")

    args = parser.parse_args(argv)

    try:
        cluster = Cluster(config=args.config)

        if args.command == "init":
            result = cluster.init(
                world_size=args.world_size,
                backend=args.backend,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
            print(json.dumps(result, indent=2))
            return 0

        if args.command == "status":
            print(json.dumps(cluster.status(), indent=2))
            return 0

        if args.command == "launch":
            cmd = cluster.launch_command(args.script, args.script_args or None)
            print(" ".join(cmd))
            return 0

        if args.command == "start":
            result = cluster.start()
            print(json.dumps(result, indent=2))
            return 0

        if args.command == "teardown":
            result = cluster.teardown()
            print(json.dumps(result, indent=2))
            return 0

    except ClusterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())