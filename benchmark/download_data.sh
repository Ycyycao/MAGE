#!/bin/bash
# Download the MemoryArena / TravelPlanner files that math, phys and travelplan read (the README's commands), skipping
# files already there. Uses HF_ENDPOINT when it is set (e.g. https://hf-mirror.com). WebShop and WebSearch have their
# own setup scripts under webshop/WebShopEnv and websearch/WebSearchEnv.
set -euo pipefail
cd "$(dirname "$0")"
HF="${HF_ENDPOINT:-https://huggingface.co}"
get() {  # get <dest> <url>
    if [ -s "$1" ]; then echo "exists: $1"; return; fi
    mkdir -p "$(dirname "$1")"
    curl -fsSL --retry 3 -o "$1.part" "$2" && mv "$1.part" "$1" && echo "downloaded: $1"
}
get math/data/data.jsonl "$HF/datasets/ZexueHe/memoryarena/resolve/main/formal_reasoning_math/data.jsonl"
get phys/data/data.jsonl "$HF/datasets/ZexueHe/memoryarena/resolve/main/formal_reasoning_phys/data.jsonl"
get travelplan/data/multitraveler.jsonl "$HF/datasets/ZexueHe/memoryarena/resolve/main/group_travel_planner/data.jsonl"
get travelplan/data/train.csv "$HF/datasets/osunlp/TravelPlanner/resolve/main/train.csv"
