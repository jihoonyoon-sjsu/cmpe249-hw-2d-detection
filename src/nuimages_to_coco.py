"""Convert nuImages val (keyframe 2D boxes) to a COCO-format GT file using COCO class ids.

Only classes with a clear COCO counterpart are evaluated. A few nuImages classes are
kept as "ignore" regions (iscrowd=1) so detections on them are neither TP nor FP.
"""
import argparse
import json
import os
from collections import Counter

COCO_CATS = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

EVAL_MAP = {
    "human.pedestrian.adult": 1,
    "human.pedestrian.child": 1,
    "human.pedestrian.construction_worker": 1,
    "human.pedestrian.police_officer": 1,
    "vehicle.bicycle": 2,
    "vehicle.car": 3,
    "vehicle.emergency.police": 3,
    "vehicle.motorcycle": 4,
    "vehicle.bus.bendy": 6,
    "vehicle.bus.rigid": 6,
    "vehicle.truck": 8,
    "vehicle.construction": 8,
    "vehicle.emergency.ambulance": 8,
}

IGNORE_MAP = {
    "human.pedestrian.personal_mobility": 1,
    "human.pedestrian.stroller": 1,
    "human.pedestrian.wheelchair": 1,
    "vehicle.trailer": 8,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/data/nuimages"))
    ap.add_argument("--version", default="v1.0-val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = os.path.join(args.root, args.version)
    load = lambda n: json.load(open(os.path.join(d, n + ".json")))
    cats = {c["token"]: c["name"] for c in load("category")}
    sample_data = {s["token"]: s for s in load("sample_data")}
    key_tokens = [s["key_camera_token"] for s in load("sample")]

    images, sd_to_id = [], {}
    for i, tok in enumerate(sorted(key_tokens, key=lambda t: sample_data[t]["filename"]), 1):
        sd = sample_data[tok]
        images.append({"id": i, "file_name": sd["filename"], "width": sd["width"], "height": sd["height"]})
        sd_to_id[tok] = i

    anns, used, dropped = [], Counter(), Counter()
    for a in load("object_ann"):
        if a["sample_data_token"] not in sd_to_id:
            continue
        name = cats[a["category_token"]]
        if name in EVAL_MAP:
            cid, crowd = EVAL_MAP[name], 0
        elif name in IGNORE_MAP:
            cid, crowd = IGNORE_MAP[name], 1
        else:
            dropped[name] += 1
            continue
        x1, y1, x2, y2 = a["bbox"]
        w, h = x2 - x1, y2 - y1
        anns.append({"id": len(anns) + 1, "image_id": sd_to_id[a["sample_data_token"]], "category_id": cid,
                     "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": crowd})
        used[(name, COCO_CATS[cid], "ignore" if crowd else "eval")] += 1

    out = {"images": images, "annotations": anns,
           "categories": [{"id": k, "name": v} for k, v in COCO_CATS.items()]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))

    print(f"images={len(images)} annotations={len(anns)}")
    for (src, dst, kind), n in sorted(used.items()):
        print(f"  {src:40s} -> {dst:10s} {kind:6s} {n}")
    for src, n in sorted(dropped.items()):
        print(f"  {src:40s} -> (not evaluated)   {n}")


if __name__ == "__main__":
    main()
