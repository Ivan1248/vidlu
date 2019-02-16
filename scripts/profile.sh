#!/bin/bash
#stdbuf -o 0 python -m cProfile -s cumtime "$@" |& tee profile.out
stdbuf -o 0 python -m cProfile -s cumtime "$@"
