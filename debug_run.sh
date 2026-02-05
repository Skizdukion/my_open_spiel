#!/usr/bin/env bash

BIN="$1"
shift

# Split remaining arguments like a real shell command line
eval exec "$BIN" $@