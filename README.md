# Animal_Detection_AI

Final project for CAP 4630 – Intro to Artificial Intelligence.  
This repository implements an animal detection pipeline using:

- **YOLOv8s** (pretrained on COCO) for object detection.
- **MobileNetV2 (TensorFlow + TFLite)** as a lightweight binary classifier (animal vs non-animal).

A small, ready-to-use image sample is included under `docs/samples` so you can run the demo
without downloading the full COCO dataset.

---

## 1. Quickstart (Windows / PowerShell)

```powershell
# Clone & create virtual environment
git clone https://github.com/VictorGal100/Animal_Detection_AI.git
cd Animal_Detection_AI

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
2. Run the YOLOv8 demo (no big downloads needed)
A small set of COCO images is already stored in docs/samples/.

powershell
Copy code
# From repo root, with .venv activated
python src\predict.py --src docs\samples --model yolov8s.pt --conf 0.20 --imgsz 640
The script filters detections to animal classes:

bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

Annotated outputs are saved under runs/animal_pred/.

3. (Optional) Regenerate samples from COCO 2017
If you want to reproduce our small animal-only dataset, you can download COCO
val2017 and build a new subset:

powershell
Copy code
# Create folders
md data\coco2017\images -ea 0
md data\coco2017\annotations -ea 0

# Download COCO val2017 images + annotations
curl -L http://images.cocodataset.org/zips/val2017.zip -o val2017.zip
curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip

# Extract
tar -xf val2017.zip -C data\coco2017\images
tar -xf annotations_trainval2017.zip -C data\coco2017

# Build a small animal-only sample set and run detection
python scripts\select_coco_animals.py --coco_root data\coco2017 --out docs\samples --n 25
python src\predict.py --src docs\samples --model yolov8s.pt --conf 0.20 --imgsz 640
The extra --classes 14 15 16 17 18 19 20 21 22 23 argument is not required
anymore because predict.py already filters animal classes internally.

4. MobileNetV2 (TensorFlow + TFLite) experiments
We also trained a custom MobileNetV2 classifier (animal vs non-animal) and
exported it to TFLite. All related files live in:

experiments/tf_mobilenetv2/AnimalAiMain.py – training script (Keras).

experiments/tf_mobilenetv2/export_tflite.py – exports .keras → .tflite.

experiments/tf_mobilenetv2/bench_tf.py – TF/CPU micro-benchmark.

experiments/tf_mobilenetv2/bench_tflite.py – TFLite micro-benchmark.

experiments/tf_mobilenetv2/benchmark.md – summary table with results.

experiments/tf_mobilenetv2/results_tf.csv and results_tflite.csv – raw logs.

Note: In our setup we used a separate virtual environment for TensorFlow (e.g. .venv-tf)
to keep dependencies cleaner.

Example commands (from repo root):

powershell
Copy code
# TensorFlow / CPU benchmark
.\.venv-tf\Scripts\python .\experiments\tf_mobilenetv2\bench_tf.py

# TFLite benchmark (XNNPACK)
.\.venv-tf\Scripts\python .\experiments\tf_mobilenetv2\bench_tflite.py
5. Mini benchmark (docs/samples, n = 53 images)
Average latency per image on our Windows CPU:

Model	Avg ms/img	Images	Notes
YOLOv8s	99.4	53	imgsz=640, conf=0.20
MobileNetV2 (TF/CPU)	59.5	53	224px, binary animal
MobileNetV2 (TFLite)	5.5	53	224px, FP32 frozen

The exact numbers are stored in experiments/tf_mobilenetv2/benchmark.md.

6. Dataset and references
Primary dataset: COCO 2017 – Common Objects in Context
Website: https://cocodataset.org/

We use a small animal-only subset from val2017, and a fixed snapshot is
included under docs/samples/ for reproducible demos.

7. Presentation video
The final presentation (slides + demo) will be available at:

(Link=https://www.youtube.com/watch?v=Y5uiCoo9Yxw)
