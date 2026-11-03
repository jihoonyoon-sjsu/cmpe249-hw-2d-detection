#!/bin/bash
# Usage: bash run_all.sh coco|nuimages [extra run_detector.py args]
set -e
DATASET=$1; shift
source ~/miniforge3/etc/profile.d/conda.sh
conda activate det
cd ~/work/hw2d
mkdir -p weights logs results

if [ "$DATASET" = coco ]; then
  GT=~/data/coco/annotations/instances_val2017.json; IMG=~/data/coco/val2017
else
  GT=~/work/hw2d/data/nuimages_val_coco6.json; IMG=~/data/nuimages
fi
SIX=1,2,3,4,6,8

for M in yolo11m rtdetr-l fasterrcnn_r50_fpn_v2; do
  OUT=results/$DATASET/$M
  LOG=logs/${DATASET}_${M}_$(date +%Y%m%d_%H%M).log
  echo "+ python src/run_detector.py --model $M --gt $GT --img-root $IMG --out $OUT $*" | tee $LOG
  python src/run_detector.py --model $M --gt $GT --img-root $IMG --out $OUT "$@" 2>&1 | tee -a $LOG
  python src/evaluate.py --gt $GT --dets $OUT/detections.json --out $OUT/metrics.json 2>&1 | tee -a $LOG
  if [ "$DATASET" = coco ]; then
    python src/evaluate.py --gt $GT --dets $OUT/detections.json --out $OUT/metrics_6cls.json --cats $SIX 2>&1 | tee -a $LOG
  fi
done
