#!/bin/bash

# Usage: bash docker.sh <command> [arguments...]
# This script performs the following:
# 1. Look for x/Dockerfile, where x is an ancestor directory.
# 2. Build the x/Dockerfile if it is found.
# 3. Execute `docker run` with:
#   - the current user's ID
#   - forwarding environment variables starting with "CUDA_" and "VIDLU_"
#   - mounting ~ of the host to /app in the container
#   - setting the container working directory relative to app/ to the host working directory relative to ~
#   - limiting resources
#   - running the provided command
#
# The container is named after the directory that contains the Docker file (x)
# with appended "-dev".
#
# Links:
# - https://blog.gougousis.net/file-permissions-the-painful-side-of-docker/

set -e

find_in_ancestor() {
  # Modified from: https://www.npmjs.com/package/find-config#algorithm
  # If X/file.ext exists, return ancestor directory X. Otherwise, return null.
  if [ -f "$1" ]; then
    echo "${PWD%/}"
  elif [ "$PWD" = / ]; then
    echo null
  else
    (cd .. && find_in_ancestor "$1")
  fi
}

dockerdir="$(find_in_ancestor Dockerfile)"
if [ "$dockerdir" = "null" ]; then  # Check if the output is "null"
  echo "Error: Dockerfile not found in ancestors of $PWD."
  exit 1
fi

printf "\nFound $dockerdir/Dockerfile\n\n"

cd $dockerdir
projname=$(basename "$dockerdir" | tr '[:upper:]' '[:lower:]')
image_name="${projname,,}-devel"  # convert to lowercase and append -dev

dockerfile_mtime=$(stat -c %Y Dockerfile 2>/dev/null || echo 0)
timestamp_file=".dockerfile_last_build"
last_build_time=$(cat $timestamp_file 2>/dev/null || echo 0)
if [[ $dockerfile_mtime -gt $last_build_time ]]; then
  build_cmd="docker build -t $image_name ."
  echo "Dockerfile changed. Building image with command: $build_cmd"
  eval $build_cmd
  echo $dockerfile_mtime > $timestamp_file
else
  echo "Dockerfile unchanged. Skipping build."
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

echo "Running command in docker: $command"
# other arguments:
# --gpus '"device=2,3"'
# --cpus=16 --memory=128G --shm-size=32G
# --env-file <(env | grep -E 'CUDA_.*|VIDLU_.*|QT.*') \
#  --network=host \
# -v "$(echo ~)/.config:$(echo ~)/.config" \

set -o xtrace  # print commands before executing them
docker run -it --rm --runtime=nvidia --gpus all \
  --shm-size=16G \
  -e "HOME=$(echo ~)" \
  -v "$(echo ~/data):$(echo ~/data)" \
  -v "$(echo ~/projects):$(echo ~/projects)" \
  --workdir $PWD \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e "TERM=xterm-256color" \
  -e "DISPLAY" \
  -v "/mnt:/mnt" \
  $image_name "$@"
set +o xtrace

#bash -c "groupadd -g $(id -g) $(id -g -n) && useradd $(id -u -n) -u $(id -u) -g $(id -g) && $command"

# TODO: bashrc and python alias
