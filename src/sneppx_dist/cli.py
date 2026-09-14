import argparse

from sneppx_dist.cluster import Cluster


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sneppx-dist", description="distributed training CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="initialize a training cluster config")
    parser_status = sub.add_parser("status", help="cluster status")
    parser_status.add_argument("--config", default="sneppx-dist.json")
    parser_teardown = sub.add_parser("teardown", help="tear down a cluster")
    parser_teardown.add_argument("--config", default="sneppx-dist.json")
    args = parser.parse_args(argv)

    if args.command == "init":
        print(Cluster().init())
    elif args.command == "status":
        print(Cluster(config=args.config).status())
    elif args.command == "teardown":
        print(Cluster(config=args.config).teardown())


if __name__ == "__main__":
    main()
