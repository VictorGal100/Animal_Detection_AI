**Summary**
Adds a simple MobileNetV2 binary classifier (animal vs non-animal), exports it to TFLite, and includes a mini benchmark against YOLOv8s and TF/CPU on docs/samples.

**What changed**
- experiments/tf_mobilenetv2/AnimalAiMain.py: training script (Keras, MobileNetV2 backbone), saves .keras
- experiments/tf_mobilenetv2/export_tflite.py: robust TFLite conversion (falls back to freezing a ConcreteFunction)
- experiments/tf_mobilenetv2/bench_tf.py: TF/CPU micro-benchmark
- experiments/tf_mobilenetv2/bench_tflite.py: TFLite micro-benchmark (XNNPACK enabled)
- .gitignore: ignores .keras, __pycache__, *.tflite, dataset folders, and temp export dirs

**Mini benchmark on my Windows CPU**
| Model                 | Avg ms/img | Images | Notes                |
|----------------------|-----------:|-------:|----------------------|
| YOLOv8s              | 99.4       | 53     | imgsz=640, conf=0.20 |
| MobileNetV2 (TF/CPU) | 59.5       | 53     | 224px, binary        |
| MobileNetV2 (TFLite) | 5.4        | 53     | FP32 frozen, XNNPACK |

**How to reproduce**
`ps1
# from repo root
.\.venv\Scripts\python .\src\predict.py --src .\docs\samples --model yolov8s.pt --conf 0.20 --imgsz 640
.\.venv-tf\Scripts\python .\experiments\tf_mobilenetv2\bench_tf.py
.\.venv-tf\Scripts\python .\experiments\tf_mobilenetv2\bench_tflite.py

Notes

Direct Keras->TFLite via MLIR failed with a ReadVariableOp missing-attribute error. The converter now falls back to freezing a ConcreteFunction and converts fine.

Large binaries are ignored by git (*.keras, *.tflite). If needed, we can publish them in a GitHub Release.

Class mapping uses train_data.class_indices; the demo prints the predicted class using that mapping to avoid label flips.
