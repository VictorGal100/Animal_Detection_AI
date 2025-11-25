import os, time, glob, csv
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL   = os.path.join("experiments","tf_mobilenetv2","mobilenetv2_animals.tflite")
SRC_DIR = os.path.join("docs","samples")
OUT_CSV = os.path.join("experiments","tf_mobilenetv2","results_tflite.csv")

if not os.path.exists(MODEL):
    raise FileNotFoundError(MODEL)

# XNNPACK suele activarse automáticamente; esto mantiene el código simple.
interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
ih, iw = input_details[0]["shape"][1:3]
idtype = input_details[0]["dtype"]

paths = []
for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
    paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))
if not paths:
    raise RuntimeError(f"No images found in {SRC_DIR}")

rows, tms = [], []
for p in paths:
    img = Image.open(p).convert("RGB").resize((iw, ih))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0).astype(idtype)
    interpreter.set_tensor(input_details[0]["index"], x)
    t0 = time.perf_counter()
    interpreter.invoke()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    tms.append(dt_ms)
    y = interpreter.get_tensor(output_details[0]["index"])
    score = float(np.squeeze(y).max())  # binario o multi-clase
    rows.append([os.path.basename(p), f"{dt_ms:.3f}", f"{score:.6f}"])

avg_ms = sum(tms) / len(tms)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["filename","ms","score"]); w.writerows(rows)

print(f"TFLite avg_ms={avg_ms:.1f}  n={len(paths)}  -> {os.path.abspath(OUT_CSV)}")