#!/bin/bash
#stdbuf -o 0 python -m cProfile -s cumtime "$@" |& tee profile.out
#stdbuf -o 0 python -m cProfile -s cumtime "$@" > profile.txt  # cumtime includes sub-calls
#~/.local/lib/python3.8/site-packages/profile_viewer
#stdbuf -o 0 python -m cProfile -s cumtime "$@"
stdbuf -o 0 python -m cProfile -s tottime -o profile2.out "$@"
snakeviz profile.out  # pip install --user snakeviz