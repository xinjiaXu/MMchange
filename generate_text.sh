#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd} TINYLLAVA_MODEL_DIR\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset_root="$data_root/LEVIR_CD"; image_a="A"; image_b="B" ;;
  levircdplus) dataset_root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a="A"; image_b="B" ;;
  cdd) dataset_root="$data_root/CDD"; image_a="A"; image_b="B" ;;
  whucd) dataset_root="$data_root/WHUCD"; image_a="A"; image_b="B" ;;
  sysucd) dataset_root="$data_root/SYSU_CD"; image_a="time1"; image_b="time2" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

model_path="$(realpath "$2")"
if [[ ! -d "$model_path" ]]; then
  printf 'TinyLLaVA model directory not found: %s\n' "$model_path" >&2
  exit 1
fi
for split in train val test; do
  for folder in "$image_a" "$image_b"; do
    if [[ ! -d "$dataset_root/$split/$folder" ]]; then
      printf 'Image directory missing: %s\n' "$dataset_root/$split/$folder" >&2
      exit 1
    fi
  done
done

work_dir="$repo_root/MMchange_work_dirs/$1"
mkdir -p "$work_dir"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"
python text_obtain.py \
  --dataset-root "$dataset_root" \
  --model-path "$model_path" \
  --output-json "$work_dir/captions.json" \
  --image-dir-a "$image_a" \
  --image-dir-b "$image_b"
