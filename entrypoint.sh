#!/bin/sh
set -e
MP="${MODEL_PATH:-/tmp/models/mobilenet_brain_tumor_classifier.h5}"
if [ ! -f "$MP" ]; then
  D="$(dirname "$MP")"
  mkdir -p "$D"
  if [ -z "$MODEL_URL" ]; then
    echo "MODEL_URL not set" >&2
    exit 1
  fi
  curl -fsSL "$MODEL_URL" -o "$MP"
fi
exec python server1.py
