"""COCO-style evaluation: summary stats, per-class AP, and per-class AP by object size."""
import argparse
import contextlib
import io
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
AREAS = ["all", "small", "medium", "large"]


def mean_valid(x):
    x = x[x > -1]
    return float(x.mean()) if x.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cats", default=None, help="comma-separated COCO category ids to restrict evaluation to")
    args = ap.parse_args()

    gt = COCO(args.gt)
    cat_ids = [int(c) for c in args.cats.split(",")] if args.cats else sorted(gt.getCatIds())
    dets = [d for d in json.load(open(args.dets)) if d["category_id"] in set(cat_ids)]
    dt = gt.loadRes(dets)

    ev = COCOeval(gt, dt, "bbox")
    ev.params.catIds = cat_ids
    ids_file = os.path.join(os.path.dirname(args.dets), "image_ids.json")
    if os.path.exists(ids_file):
        ev.params.imgIds = json.load(open(ids_file))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    print(buf.getvalue())

    prec = ev.eval["precision"]  # [T, R, K, A, M]
    per_class = {}
    for k, cid in enumerate(cat_ids):
        name = gt.loadCats(cid)[0]["name"]
        row = {"num_gt": len(gt.getAnnIds(imgIds=ev.params.imgIds, catIds=[cid], iscrowd=False))}
        for a, area in enumerate(AREAS):
            row[f"AP_{area}"] = mean_valid(prec[:, :, k, a, -1])
        row["AP50"] = mean_valid(prec[0, :, k, 0, -1])
        row["AP75"] = mean_valid(prec[5, :, k, 0, -1])
        per_class[name] = row

    out = {"summary": dict(zip(STAT_NAMES, map(float, ev.stats))), "per_class": per_class,
           "cat_ids": cat_ids, "num_dets": len(dets)}
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
