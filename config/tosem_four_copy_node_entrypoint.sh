#!/bin/sh
set -eu
# Docker gives this image a hard limit of 524288; raise the soft limit before
# starting systemd/containerd so kubelet and pod containers inherit it.
ulimit -n 524288 2>/dev/null || ulimit -n 65536
exec /usr/local/bin/entrypoint "$@"
