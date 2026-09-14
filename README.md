# SNEPPX Dist - Distributed Training CLI

`<project> dis` tooling: `sneppx-dist init`, `status`, `teardown` for
launching NCCL/DDP/ZeRO jobs on a multi-GPU box.

> Status: skeleton (WIP)

## Layout
- `src/sneppx_dist/cli.py` - subcommand dispatcher
- `src/sneppx_dist/cluster.py` - cluster/launch skeleton
- `tests/` - smoke tests

## Roadmap
- [ ] `init` (detect GPUs, write config)
- [ ] job launch (torchrun/NCCL)
- [ ] health/status polling
- [ ] enterprise dashboard (paid tier)

## License
MIT - part of the SneppX open-core ecosystem.
