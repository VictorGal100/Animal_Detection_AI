# -*- coding: utf-8 -*-
import os, glob, random, time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import load_img, img_to_array

HERE = os.path.dirname(__file__)
IMG_ROOT = os.path.join(HERE, "Img")

# 1. Load dataset
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
)

"""change to the image folder"""

train_data = datagen.flow_from_directory(
    IMG_ROOT,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=42
)

"""change to the image folder"""
val_data = datagen.flow_from_directory(
    IMG_ROOT,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# 2. Build pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train model
history = model.fit(train_data, validation_data=val_data, epochs=5)


# 4. Predict a sample image
candidates = glob.glob(os.path.join(IMG_ROOT, "Animal", "*.jpg")) + \
             glob.glob(os.path.join(IMG_ROOT, "Non-Animal", "*.jpg"))
sample_image_path = random.choice(candidates)

img = load_img(sample_image_path, target_size=(224, 224))
x = img_to_array(img) / 255.0
t0 = time.time()
p = float(model.predict(np.expand_dims(x, 0), verbose=0)[0][0])
dt = (time.time() - t0) * 1000

idx_to_name = {v: k for k, v in train_data.class_indices.items()}  # {0:"Animal",1:"Non-Animal"} (alfabético)
prob_animal = p if idx_to_name[1] == "Animal" else (1.0 - p)
pred_class = "Animal" if prob_animal > 0.5 else "Non-Animal"

print(f"[DEMO] {os.path.basename(sample_image_path)} -> prob_animal={prob_animal:.3f}  ({dt:.1f} ms)")
print(f"[DEMO] Predicción: {pred_class}")
