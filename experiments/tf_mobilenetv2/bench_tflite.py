import os, time, glob
import numpy as np
from tensorflow import lite as tflite
from tensorflow.keras.utils import load_img, img_to_array

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = os.path.join(HERE, "mobilenetv2_animals.tflite")
SAMPLES = os.path.join(ROOT, "docs", "samples")
OUTCSV = os.path.join(HERE, "results_tflite.csv")

# Collect sample images
paths = sorted(glob.glob(os.path.join(SAMPLES, "*.jpg")))
if not paths:
    raise FileNotFoundError(f"No JPGs found in: {SAMPLES}")

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

H, W = input_info["shape"][1], input_info["shape"][2]
in_dtype = input_info["dtype"]

def preprocess(img_path):
    img = load_img(img_path, target_size=(H, W))
    x = img_to_array(img)
    if in_dtype == np.float32:
        x = x / 255.0  # match training rescale
    x = np.expand_dims(x.astype(in_dtype), 0)
    return x

# Warm-up
x0 = preprocess(paths[0])
for _ in range(3):
    interpreter.set_tensor(input_info["index"], x0)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_info["index"])

# Timed runs
times_ms = []
with open(OUTCSV, "w", encoding="utf-8") as f:
    f.write("filename,ms,score\n")
    for p in paths:
        x = preprocess(p)
        t0 = time.perf_counter()
        interpreter.set_tensor(input_info["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_info["index"])
        ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(ms)
        # y is sigmoid score for class 1 (usually "Non-Animal")
        score = float(y.flatten()[0])
        f.write(f"{os.path.basename(p)},{ms:.3f},{score:.6f}\n")

avg = float(np.mean(times_ms))
print(f"TFLite avg_ms={avg:.1f}  n={len(times_ms)}  -> {OUTCSV}")