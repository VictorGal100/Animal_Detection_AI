# -*- coding: utf-8 -*-
import os, glob, time, csv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "mobilenetv2_animals.keras")
SAMPLES = os.path.join(HERE, "..", "..", "docs", "samples")

m = load_model(MODEL)
paths = sorted(glob.glob(os.path.join(SAMPLES, "*.jpg")))
out_csv = os.path.join(HERE, "results_tf.csv")

times = []
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename","prob_animal","ms"])
    for pth in paths:
        x = img_to_array(load_img(pth, target_size=(224,224)))/255.0
        t0 = time.time()
        prob = float(m.predict(np.expand_dims(x,0), verbose=0)[0][0])
        ms = (time.time()-t0)*1000
        times.append(ms)
        w.writerow([os.path.basename(pth), prob, f"{ms:.1f}"])
avg_ms = sum(times)/len(times) if times else 0
print(f"TF avg_ms={avg_ms:.1f}  n={len(times)}  -> {out_csv}")
