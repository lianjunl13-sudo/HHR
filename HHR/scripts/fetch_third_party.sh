#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/thirdparty"

clone_at() {
  local url=$1
  local revision=$2
  local destination=$3
  if [[ -d "$destination" ]]; then
    return
  fi
  git clone --depth 1 --branch "$revision" "$url" "$destination"
}

clone_at https://github.com/NVIDIA/cutlass.git v3.6.0 "$ROOT/thirdparty/cutlass"
clone_at https://github.com/rapidsai/raft.git branch-23.04 "$ROOT/thirdparty/raft"
clone_at https://github.com/rapidsai/rmm.git branch-22.04 "$ROOT/thirdparty/rmm"
clone_at https://github.com/gabime/spdlog.git v1.8.5 "$ROOT/thirdparty/spdlog"
