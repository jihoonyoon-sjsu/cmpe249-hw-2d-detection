"""Cross-dataset analysis: tables, figures, and failure analysis on nuImages."""
import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["yolo11m", "rtdetr-l", "fasterrcnn_r50_fpn_v2"]
LABEL = {"yolo11m": "YOLO11m", "rtdetr-l": "RT-DETR-L", "fasterrcnn_r50_fpn_v2": "Faster R-CNN R50-FPN v2"}
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
COCO_ID = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "bus": 6, "truck": 8}
ID_NAME = {v: k for k, v in COCO_ID.items()}
HEIGHT_BINS = [0, 16, 32, 64, 128, 256, 10000]


def load(p):
    return json.load(open(p))


def iou_matrix(a, b):
    """a: [N,4] xywh, b: [M,4] xywh -> IoU [N,M] and intersection-over-area-of-a [N,M]."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None]
    return inter / (area_a + area_b - inter + 1e-9), inter / (area_a + 1e-9)


def nuimages_full_gt(root, version, coco6):
    """All nuImages boxes (every category) keyed by our COCO image id, plus rider attribute."""
    d = os.path.join(root, version)
    cats = {c["token"]: c["name"] for c in load(os.path.join(d, "category.json"))}
    attrs = {a["token"]: a["name"] for a in load(os.path.join(d, "attribute.json"))}
    fname_to_id = {im["file_name"]: im["id"] for im in coco6["images"]}
    sd_to_id = {s["token"]: fname_to_id[s["filename"]] for s in load(os.path.join(d, "sample_data.json"))
                if s["filename"] in fname_to_id}
    per_img = defaultdict(list)
    for a in load(os.path.join(d, "object_ann.json")):
        if a["sample_data_token"] not in sd_to_id:
            continue
        x1, y1, x2, y2 = a["bbox"]
        per_img[sd_to_id[a["sample_data_token"]]].append(
            (cats[a["category_token"]], [x1, y1, x2 - x1, y2 - y1], {attrs[t] for t in a["attribute_tokens"]}))
    return per_img


def table_rows(res):
    rows = []
    for m in MODELS:
        c, n = res["coco6"][m]["summary"], res["nuim"][m]["summary"]
        full = res["coco80"][m]["summary"]
        rows.append([LABEL[m], full["AP"], c["AP"], n["AP"], n["AP"] - c["AP"], c["AP50"], n["AP50"], c["AP75"], n["AP75"],
                     c["AP_small"], n["AP_small"], c["AP_medium"], n["AP_medium"], c["AP_large"], n["AP_large"],
                     c["AR100"], n["AR100"]])
    return rows


def fig_per_class(res, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    x = np.arange(len(CLASSES))
    for ax, m in zip(axes, MODELS):
        c = [100 * res["coco6"][m]["per_class"][k]["AP_all"] for k in CLASSES]
        n = [100 * res["nuim"][m]["per_class"][k]["AP_all"] for k in CLASSES]
        ax.bar(x - 0.2, c, 0.4, label="COCO val2017")
        ax.bar(x + 0.2, n, 0.4, label="nuImages val")
        ax.set_xticks(x, CLASSES, rotation=30)
        ax.set_title(LABEL[m])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("AP@[.50:.95] (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_size(res, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    sizes = ["AP_small", "AP_medium", "AP_large"]
    x = np.arange(3)
    w = 0.13
    for i, m in enumerate(MODELS):
        ax.bar(x + (2 * i - 5) * w / 2, [100 * res["coco6"][m]["summary"][s] for s in sizes], w,
               label=f"{LABEL[m]} COCO", color=f"C{i}", alpha=0.45)
        ax.bar(x + (2 * i - 4) * w / 2, [100 * res["nuim"][m]["summary"][s] for s in sizes], w,
               label=f"{LABEL[m]} nuImages", color=f"C{i}")
    ax.set_xticks(x, ["small (<32²)", "medium", "large (>96²)"])
    ax.set_ylabel("AP@[.50:.95] (%)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def failure_analysis(coco6, full_gt, dets_by_model, score_thr):
    """Per model at a fixed operating point: recall by box height, class confusion, and rider FPs."""
    gt6 = defaultdict(list)
    for a in coco6["annotations"]:
        gt6[a["image_id"]].append(a)
    out = {}
    for m, dets in dets_by_model.items():
        per_img = defaultdict(list)
        for d in dets:
            if d["score"] >= score_thr:
                per_img[d["image_id"]].append(d)
        hit = np.zeros((len(HEIGHT_BINS) - 1, 2))
        confusion = Counter()
        fp_on = Counter()
        rider_total = rider_person = 0
        empty_hits = []
        for im in coco6["images"]:
            iid = im["id"]
            gts = [a for a in gt6[iid] if not a["iscrowd"]]
            ds = per_img.get(iid, [])
            db = np.array([d["bbox"] for d in ds]).reshape(-1, 4)
            dc = np.array([d["category_id"] for d in ds])
            if gts:
                gb = np.array([a["bbox"] for a in gts])
                gc = np.array([a["category_id"] for a in gts])
                if len(ds):
                    iou, _ = iou_matrix(gb, db)
                    same = iou * (gc[:, None] == dc[None])
                    found = same.max(1) >= 0.5
                else:
                    iou = np.zeros((len(gts), 0))
                    found = np.zeros(len(gts), bool)
                h_bin = np.digitize(gb[:, 3], HEIGHT_BINS) - 1
                for b, f in zip(h_bin, found):
                    hit[b, 0] += f
                    hit[b, 1] += 1
                for g in range(len(gts)):
                    if found[g] or not len(ds):
                        continue
                    j = iou[g].argmax()
                    if iou[g, j] >= 0.5:
                        confusion[(ID_NAME[gc[g]], ID_NAME[int(dc[j])])] += 1
            # Detections that match no evaluated GT: what raw nuImages object (if any) do they sit on?
            if len(ds):
                raw = full_gt.get(iid, [])
                rb = np.array([r[1] for r in raw]).reshape(-1, 4)
                if gts:
                    giou, _ = iou_matrix(db, np.array([a["bbox"] for a in gts]))
                    matched = (giou * (dc[:, None] == np.array([a["category_id"] for a in gts])[None])).max(1) >= 0.5
                else:
                    matched = np.zeros(len(ds), bool)
                if len(raw):
                    riou, _ = iou_matrix(db, rb)
                for j in np.where(~matched & np.isin(dc, list(COCO_ID.values())))[0]:
                    if len(raw) and riou[j].max() >= 0.5:
                        fp_on[(ID_NAME[int(dc[j])], raw[riou[j].argmax()][0])] += 1
                    elif len(raw) and riou[j].max() >= 0.1:
                        fp_on[(ID_NAME[int(dc[j])], "poor localization (0.1<=IoU<0.5)")] += 1
                    else:
                        fp_on[(ID_NAME[int(dc[j])], "no labeled object (IoU<0.1)")] += 1
                        if ds[j]["score"] >= 0.7:
                            empty_hits.append((ds[j]["score"], iid, ds[j]["bbox"], ID_NAME[int(dc[j])]))
            # Taxonomy check: nuImages cycle boxes include the rider; does the model fire "person" on the rider?
            riders = [r for r in full_gt.get(iid, []) if r[0] in ("vehicle.bicycle", "vehicle.motorcycle")
                      and "cycle.with_rider" in r[2]]
            if riders:
                rider_total += len(riders)
                persons = db[dc == 1] if len(ds) else np.zeros((0, 4))
                if len(persons):
                    _, ioa = iou_matrix(persons, np.array([r[1] for r in riders]))
                    rider_person += int((ioa.max(0) >= 0.7).sum())
        out[m] = {
            "score_thr": score_thr,
            "recall_by_height": {f"{HEIGHT_BINS[i]}-{HEIGHT_BINS[i + 1]}px": [float(hit[i, 0] / max(hit[i, 1], 1)), int(hit[i, 1])]
                                 for i in range(len(HEIGHT_BINS) - 1)},
            "confusion_top": [[g, p, n] for (g, p), n in confusion.most_common(10)],
            "unmatched_dets_top": [[p, r, n] for (p, r), n in fp_on.most_common(15)],
            "cyclists_with_rider": rider_total,
            "cyclists_with_person_det_inside": rider_person,
            "confident_empty_hits": sorted(empty_hits, reverse=True),
        }
    return out


def fig_recall_height(fa, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(MODELS):
        bins = list(fa[m]["recall_by_height"].keys())
        ax.plot(range(len(bins)), [100 * fa[m]["recall_by_height"][b][0] for b in bins], "o-", label=LABEL[m], color=f"C{i}")
    ax.set_xticks(range(len(bins)), [b.replace("-10000", "+") for b in bins])
    ax.set_xlabel("GT box height on nuImages (pixels, original 1600x900 image)")
    ax.set_ylabel(f"Recall @ IoU 0.5, score >= {fa[MODELS[0]]['score_thr']} (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_crop_gallery(hits, coco6, img_root, out, n=24, seed=0):
    """Random sample of confident detections that overlap no labeled nuImages object."""
    import cv2
    files = {im["id"]: im["file_name"] for im in coco6["images"]}
    rng = np.random.default_rng(seed)
    pick = [hits[i] for i in rng.choice(len(hits), min(n, len(hits)), replace=False)]
    cols = 6
    fig, axes = plt.subplots((len(pick) + cols - 1) // cols, cols, figsize=(2.4 * cols, 2.6 * len(pick) / cols))
    for ax, (s, iid, (x, y, w, h), name) in zip(axes.flat, pick):
        img = cv2.imread(os.path.join(img_root, files[iid]))[:, :, ::-1]
        pad = int(max(w, h) * 0.5) + 10
        x0, y0 = max(int(x) - pad, 0), max(int(y) - pad, 0)
        crop = img[y0:int(y + h) + pad, x0:int(x + w) + pad]
        ax.imshow(crop)
        ax.add_patch(plt.Rectangle((x - x0, y - y0), w, h, fill=False, color="red", lw=1.5))
        ax.set_title(f"{name} {s:.2f}  h={h:.0f}px", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_speed_accuracy(res, speed, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (ds, key, title) in zip(axes, (("coco", "coco80", "COCO val2017 (80 classes)"), ("nuimages", "nuim", "nuImages val (6 classes)"))):
        for i, m in enumerate(MODELS):
            s = speed[ds][m]
            ap = 100 * res[key][m]["summary"]["AP"]
            ax.scatter(s["latency_ms_median"], ap, s=s["peak_gpu_mem_mb"] / 2, color=f"C{i}", alpha=0.7)
            ax.annotate(f"{LABEL[m]}\n{s['peak_gpu_mem_mb']:.0f} MB", (s["latency_ms_median"], ap), fontsize=8,
                        xytext=(6, -4), textcoords="offset points")
        ax.set_xlabel("median latency per image (ms), batch 1, FP32")
        ax.set_ylabel("AP@[.50:.95] (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.margins(x=0.35, y=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.expanduser("~/work/hw2d/results"))
    ap.add_argument("--gt", default=os.path.expanduser("~/work/hw2d/data/nuimages_val_coco6.json"))
    ap.add_argument("--coco-gt", default=os.path.expanduser("~/data/coco/annotations/instances_val2017.json"))
    ap.add_argument("--nuimages-root", default=os.path.expanduser("~/data/nuimages"))
    ap.add_argument("--out", default=os.path.expanduser("~/work/hw2d/report"))
    ap.add_argument("--score-thr", type=float, default=0.3)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)

    r = args.results
    res = {"coco80": {m: load(f"{r}/coco/{m}/metrics.json") for m in MODELS},
           "coco6": {m: load(f"{r}/coco/{m}/metrics_6cls.json") for m in MODELS},
           "nuim": {m: load(f"{r}/nuimages/{m}/metrics.json") for m in MODELS}}

    fig_per_class(res, os.path.join(args.out, "figs", "per_class_ap.png"))
    if os.path.isdir(f"{r}/speed"):
        speed = {ds: {m: load(f"{r}/speed/{ds}/{m}/speed.json") for m in MODELS} for ds in ("coco", "nuimages")}
        fig_speed_accuracy(res, speed, os.path.join(args.out, "figs", "speed_accuracy.png"))
    fig_size(res, os.path.join(args.out, "figs", "ap_by_size.png"))

    ID_NAME.update({c["id"]: c["name"] for c in load(args.coco_gt)["categories"]})
    coco6 = load(args.gt)
    full_gt = nuimages_full_gt(args.nuimages_root, "v1.0-val", coco6)
    dets = {m: [d for d in load(f"{r}/nuimages/{m}/detections.json") if d["score"] >= args.score_thr] for m in MODELS}
    fa = failure_analysis(coco6, full_gt, dets, args.score_thr)
    fig_recall_height(fa, os.path.join(args.out, "figs", "recall_by_height.png"))
    for m in MODELS:
        fig_crop_gallery(fa[m]["confident_empty_hits"], coco6, args.nuimages_root,
                         os.path.join(args.out, "figs", f"empty_hits_{m}.png"))
        fa[m]["num_confident_empty_hits"] = len(fa[m].pop("confident_empty_hits"))

    summary = {"table": table_rows(res), "failure": fa,
               "per_class": {m: {"coco": res["coco6"][m]["per_class"], "nuim": res["nuim"][m]["per_class"]} for m in MODELS}}
    json.dump(summary, open(os.path.join(args.out, "analysis.json"), "w"), indent=2)
    for m in MODELS:
        f = fa[m]
        print(f"== {m}")
        print("  recall by height:", {k: round(100 * v[0], 1) for k, v in f["recall_by_height"].items()})
        print("  confusion (gt->pred):", f["confusion_top"][:6])
        print("  unmatched dets:", f["unmatched_dets_top"][:10])
        print(f"  riders: {f['cyclists_with_person_det_inside']}/{f['cyclists_with_rider']}  confident empty hits: {f['num_confident_empty_hits']}")


if __name__ == "__main__":
    main()
