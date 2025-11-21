# Alexander Garcia — Animal Detection AI (baseline demo)
# Quick script to run a pretrained YOLOv8 model on our sample images
# and keep only animal classes so the outputs look clean.

from ultralytics import YOLO
from pathlib import Path
import argparse

ANIMAL_NAMES = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe"
}

def main():
    parser = argparse.ArgumentParser(description="Animal Detection (YOLOv8, pretrained).")
    parser.add_argument("--src", default="docs/samples", help="Folder with test images/videos.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights (e.g., yolov8n.pt, yolov8s.pt).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    args = parser.parse_args()

    # 1) Load pretrained COCO model
    model = YOLO(args.model)

    # 2) Build list of class IDs for animals (from model's label map)
    animal_ids = [i for i, name in model.names.items() if name in ANIMAL_NAMES]
    if not animal_ids:
        raise RuntimeError("No animal class IDs found in this model's label map.")

    # 3) Run predictions on our samples and save annotated outputs
    out_dir = Path("runs/animal_pred")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=args.src,
        classes=animal_ids,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project="runs",
        name="animal_pred"
    )

    print(f"Done. Check {out_dir} for annotated images/videos.")
    # Optional: print a tiny summary
    num_items = sum(len(r.boxes) if getattr(r, 'boxes', None) is not None else 0 for r in results)
    print(f"Files processed: {len(results)} | Total detections (animals): {num_items}")

if __name__ == "__main__":
    main()

