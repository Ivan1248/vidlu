#!/bin/bash

# Usage: bash podman.sh <command> [arguments...]
# Podman counterpart of dock-rootless.sh. Differences from the Docker version:
# - Looks for Containerfile (Podman convention), not Dockerfile.
# - Drops --user, /etc/passwd, /etc/group: rootless Podman maps UIDs via
#   subuid/subgid so container root already resolves to the host user on disk.
# - --runtime=nvidia replaced by --gpus all (CDI-based, requires nvidia-ctk).
# - Adds --ipc=host so torch DataLoader workers can share memory.
#
# The image is named after the directory containing Containerfile, lowercased,
# with "-devel" appended.

set -e

find_in_ancestor() {
  # If X/file.ext exists, return ancestor directory X. Otherwise, return null.
  if [ -f "$1" ]; then
    echo "${PWD%/}"
  elif [ "$PWD" = / ]; then
    echo null
  else
    (cd .. && find_in_ancestor "$1")
  fi
}

containerdir="$(find_in_ancestor Containerfile)"
if [ "$containerdir" = "null" ]; then
  echo "Error: Containerfile not found in ancestors of $PWD."
  exit 1
fi

printf "\nFound $containerdir/Containerfile\n\n"

cd $containerdir
projname=$(basename "$containerdir" | tr '[:upper:]' '[:lower:]')
image_name="${projname,,}-devel"

containerfile_mtime=$(stat -c %Y Containerfile 2>/dev/null || echo 0)
timestamp_file=".containerfile_last_build"
last_build_time=$(cat $timestamp_file 2>/dev/null || echo 0)
if [[ $containerfile_mtime -gt $last_build_time ]]; then
  build_cmd="podman build -t $image_name -f Containerfile ."
  echo "Containerfile changed. Building image with command: $build_cmd"
  eval $build_cmd
  echo $containerfile_mtime > $timestamp_file
else
  echo "Containerfile unchanged. Skipping build."
fi
cd -

echo $(pwd)


command=$(awk -v q="'" '
  function shellquote(s) {
    gsub(q, q "\\" q q, s)
    return q s q
  }
  BEGIN {
    for (i = 1; i < ARGC; i++) {
      printf "%s", sep shellquote(ARGV[i])
      sep = " "
    }
    printf "\n"
  }' "$@")

echo "Running command in podman: $command"
# other arguments:
# --gpus '"device=2,3"'
# --cpus=16 --memory=128G
# --env-file <(env | grep -E 'CUDA_.*|VIDLU_.*|QT.*') \
#  --network=host \
# -v "$(echo ~)/.config:$(echo ~)/.config" \

# --- GUI forwarding (X11 / Xwayland / Wayland / session D-Bus) -------------
# Build podman args based on what's actually running on the host, so the same
# script works on X11 sessions, Wayland sessions (with or without Xwayland),
# and headless hosts (where it forwards nothing).
gui_args=()
xauth_file=""

# X11 / Xwayland: forward DISPLAY plus an X authority cookie.
if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
  xauth_file="$(mktemp --tmpdir xauth.XXXXXX)"
  # The "ffff" family-wildcard trick makes the cookie accept the connection
  # regardless of the hostname the container sees.
  xauth nlist "$DISPLAY" 2>/dev/null \
    | sed -e 's/^..../ffff/' \
    | xauth -f "$xauth_file" nmerge - 2>/dev/null || true
  gui_args+=(
    -e "DISPLAY=$DISPLAY"
    -e "XAUTHORITY=/tmp/.Xauthority"
    -v "$xauth_file:/tmp/.Xauthority:ro"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
  )
  # SSH X11 forwarding (DISPLAY like "localhost:11.0" or "host:N") routes
  # over a TCP port on the host's loopback. The container's "localhost" is
  # itself, so it needs the host network namespace to reach that port.
  case "$DISPLAY" in
    :*) ;;  # pure local socket, no TCP needed
    *)  gui_args+=( --network=host ) ;;
  esac
  # Permit local connections from this user only (narrower than `xhost +local:`).
  xhost +SI:localuser:"$(id -un)" >/dev/null 2>&1 || true
  trap 'rm -f "$xauth_file"' EXIT
fi

# Wayland: forward the compositor socket and a DRI device for GPU clients.
if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -n "${XDG_RUNTIME_DIR:-}" ] \
   && [ -S "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ]; then
  gui_args+=(
    -e "WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
    -e "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
    -v "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY"
    --device /dev/dri
  )
fi

# Session D-Bus: many GUI toolkits expect it; harmless to skip when absent.
if [ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
  bus_sock="${DBUS_SESSION_BUS_ADDRESS#unix:path=}"
  bus_sock="${bus_sock%%,*}"
  if [ -S "$bus_sock" ]; then
    gui_args+=(
      -e "DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS"
      -v "$bus_sock:$bus_sock"
    )
  fi
fi
# --------------------------------------------------------------------------

# W&B: mount ~/.netrc so a `wandb login` (on the host or in an earlier container) persists,
# and forward any WANDB_* variables exported in the calling shell for one-off overrides.
# The files are touched first so that podman does not create directories in their place.
netrc_file="$(echo ~/.netrc)"
touch "$netrc_file"
wandb_args=(-v "$netrc_file:$netrc_file")
for var in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_MODE; do
  if [ -n "${!var:-}" ]; then wandb_args+=(-e "$var=${!var}"); fi
done

# Bash history, persisted across container runs.
histfile="$(echo ~/.bash_history)"
touch "$histfile"

set -o xtrace
podman run -it --rm --gpus all \
  --ipc=host \
  -e "HOME=$(echo ~)" \
  -v "$(echo ~/data):$(echo ~/data)" \
  -v "$(echo ~/projects):$(echo ~/projects)" \
  -v "$(echo ~/.cache):$(echo ~/.cache)" \
  -v "$histfile:$histfile" \
  --workdir $PWD \
  -e "TERM=xterm-256color" \
  "${gui_args[@]}" \
  "${wandb_args[@]}" \
  -v "/mnt:/mnt" \
  $image_name "$@"
set +o xtrace
