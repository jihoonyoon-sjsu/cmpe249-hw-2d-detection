#!/bin/bash
# Clean latency / memory benchmark: GPU otherwise idle, 1000 images per dataset, batch 1, FP32.
# Usage: bash run_speed.sh
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate det
cd ~/work/hw2d
mkdir -p logs results/speed
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv | tee logs/speed_gpu_state.txt

for DATASET in coco nuimages; do
  if [ "$DATASET" = coco ]; then
    GT=~/data/coco/annotations/instances_val2017.json; IMG=~/data/coco/val2017
  else
    GT=~/work/hw2d/data/nuimages_val_coco6.json; IMG=~/data/nuimages
  fi
  for M in yolo11m rtdetr-l fasterrcnn_r50_fpn_v2; do
    OUT=results/speed/$DATASET/$M
    LOG=logs/speed_${DATASET}_${M}_$(date +%Y%m%d_%H%M).log
    echo "+ python src/run_detector.py --model $M --gt $GT --img-root $IMG --out $OUT --limit 1000 --warmup 50" | tee $LOG
    python src/run_detector.py --model $M --gt $GT --img-root $IMG --out $OUT --limit 1000 --warmup 50 2>&1 | tee -a $LOG
  done
done
