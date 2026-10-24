#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"
num_workers="${NUM_WORKERS:-8}"
max_steps="${MAX_STEPS:-40000}"
batch_size="${BATCH_SIZE:-32}"
lr="${LR:-5e-4}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset_root="$data_root/LEVIR_CD"; image_a="A"; image_b="B"; label_dir="label" ;;
  levircdplus) dataset_root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a="A"; image_b="B"; label_dir="label" ;;
  cdd) dataset_root="$data_root/CDD"; image_a="A"; image_b="B"; label_dir="OUT" ;;
  whucd) dataset_root="$data_root/WHUCD"; image_a="A"; image_b="B"; label_dir="label" ;;
  sysucd) dataset_root="$data_root/SYSU_CD"; image_a="time1"; image_b="time2"; label_dir="label" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

work_dir="$repo_root/MMchange_work_dirs/$1"
captions="${CAPTIONS_JSON:-$work_dir/captions.json}"
if [[ ! -f "$captions" ]]; then
  printf 'Caption JSON not found: %s\nRun generate_text.sh with the downloaded TinyLLaVA model first.\n' "$captions" >&2
  exit 1
fi
for split in train val test; do
  for folder in "$image_a" "$image_b" "$label_dir"; do
    if [[ ! -d "$dataset_root/$split/$folder" ]]; then
      printf 'Dataset directory missing: %s\nRun prepare_server_data.sh first or set DATA_ROOT.\n' "$dataset_root/$split/$folder" >&2
      exit 1
    fi
  done
done

mkdir -p "$work_dir"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"
python train.py \
  --dataset-root "$dataset_root" \
  --dataset-name "$1" \
  --captions-json "$captions" \
  --image-dir-a "$image_a" \
  --image-dir-b "$image_b" \
  --label-dir "$label_dir" \
  --output-dir "$repo_root/MMchange_work_dirs" \
  --num_workers "$num_workers" \
  --max_steps "$max_steps" \
  --batch_size "$batch_size" \
  --lr "$lr"
