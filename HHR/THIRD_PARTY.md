# Third-party dependencies

The CUDA extension expects the following projects under `thirdparty/`:

| Directory | Upstream project | Version used by the reference build |
| --- | --- | --- |
| `cutlass` | `https://github.com/NVIDIA/cutlass` | `v3.6.0` |
| `raft` | `https://github.com/rapidsai/raft` | `branch-23.04` |
| `rmm` | `https://github.com/rapidsai/rmm` | `branch-22.04` |
| `spdlog` | `https://github.com/gabime/spdlog` | `v1.8.5` |

Each dependency remains governed by its upstream license. The fetch script
clones only the selected revision and does not copy local repository metadata.
