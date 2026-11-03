"""Run a COCO-pretrained detector over a COCO-format image set.

Saves COCO-format detections plus latency / GPU-memory stats. Timing covers
decoded image in memory -> final boxes on CPU (preprocess + forward + postprocess/NMS),
batch size 1, FP32, after a warmup.
"""
import argparse
import json
import os
import platform
import time

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                85, 86, 87, 88, 89, 90]


class Images(Dataset):
    def __init__(self, coco, img_root, limit=None):
        self.ids = sorted(coco.getImgIds())[:limit]
        self.coco, self.root = coco, img_root

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        info = self.coco.loadImgs(self.ids[i])[0]
        return self.ids[i], cv2.imread(os.path.join(self.root, info["file_name"]))


def keep_numpy(sample):
    return sample


class UltralyticsDetector:
    def __init__(self, kind, weights, imgsz, conf, iou, max_det):
        from ultralytics import RTDETR, YOLO
        self.model = (RTDETR if kind == "rtdetr" else YOLO)(weights)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=0, half=False, verbose=False)

    def __call__(self, bgr):
        r = self.model.predict(bgr, **self.kw)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), np.array(COCO80_TO_91)[b.cls.cpu().numpy().astype(int)]


class TorchvisionDetector:
    def __init__(self, conf, max_det):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                                box_score_thresh=conf, box_detections_per_img=max_det)
        self.model.eval().cuda()

    @torch.inference_mode()
    def __call__(self, bgr):
        x = torch.from_numpy(bgr[:, :, ::-1].copy()).cuda().permute(2, 0, 1).float().div_(255)
        o = self.model([x])[0]
        return o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy()


def build(args):
    if args.model == "yolo11m":
        return UltralyticsDetector("yolo", os.path.join(args.weights_dir, "yolo11m.pt"), args.imgsz, args.conf, args.iou, args.max_det)
    if args.model == "rtdetr-l":
        return UltralyticsDetector("rtdetr", os.path.join(args.weights_dir, "rtdetr-l.pt"), args.imgsz, args.conf, args.iou, args.max_det)
    if args.model == "fasterrcnn_r50_fpn_v2":
        return TorchvisionDetector(args.conf, args.max_det)
    raise ValueError(args.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolo11m", "rtdetr-l", "fasterrcnn_r50_fpn_v2"])
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--weights-dir", default=os.path.expanduser("~/work/hw2d/weights"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    coco = COCO(args.gt)
    loader = DataLoader(Images(coco, args.img_root, args.limit), batch_size=None, num_workers=4, collate_fn=keep_numpy)
    det = build(args)

    first = next(iter(loader))[1]
    for _ in range(args.warmup):
        det(first)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    raw, times = [], []
    t_start = time.time()
    for n, (img_id, bgr) in enumerate(loader, 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        boxes, scores, labels = det(bgr)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        raw.append((img_id, boxes, scores, labels))
        if n % 1000 == 0:
            print(f"{n}/{len(loader)} images, mean {np.mean(times):.1f} ms", flush=True)

    results = []
    for img_id, boxes, scores, labels in raw:
        for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            results.append({"image_id": int(img_id), "category_id": int(c),
                            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                            "score": round(s, 5)})

    t = np.array(times)
    stats = {
        "model": args.model, "gt": args.gt, "num_images": len(t), "imgsz": args.imgsz if args.model != "fasterrcnn_r50_fpn_v2" else "min800/max1333",
        "conf": args.conf, "nms_iou": args.iou if args.model == "yolo11m" else ("n/a (NMS-free)" if args.model == "rtdetr-l" else 0.5),
        "max_det": args.max_det, "precision": "fp32", "batch": 1,
        "latency_ms_mean": float(t.mean()), "latency_ms_median": float(np.median(t)), "latency_ms_p90": float(np.percentile(t, 90)),
        "fps_mean": float(1000 / t.mean()), "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved() / 2**20, "wall_time_s": time.time() - t_start,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version(),
    }
    json.dump(results, open(os.path.join(args.out, "detections.json"), "w"))
    json.dump(loader.dataset.ids, open(os.path.join(args.out, "image_ids.json"), "w"))
    json.dump(stats, open(os.path.join(args.out, "speed.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
