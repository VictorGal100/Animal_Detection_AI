# export_tflite.py
import os, sys, argparse, subprocess, numpy as np, tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

HERE = os.path.dirname(__file__)
KERAS_PATH = os.path.join(HERE, "mobilenetv2_animals.keras")
TFLITE_PATH = os.path.join(HERE, "mobilenetv2_animals.tflite")

def save_bytes(tfl_bytes: bytes):
    with open(TFLITE_PATH, "wb") as f:
        f.write(tfl_bytes)
    sz_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"[child] OK -> {TFLITE_PATH}  ({sz_mb:.2f} MB)")

def child_convert(mode: str) -> int:
    print(f"[child] mode={mode}")
    print(f"[child] Loading Keras model: {KERAS_PATH}")
    model = tf.keras.models.load_model(KERAS_PATH)
    _ = model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)

    try:
        if mode == "E_freeze_fn_builtins":
            # Build a concrete function with variables frozen to constants
            @tf.function(input_signature=[tf.TensorSpec([1,224,224,3], tf.float32, name="input")])
            def serving(x):
                y = model(x, training=False)
                y = tf.identity(y, name="output")
                return y

            concrete = serving.get_concrete_function()
            frozen = convert_variables_to_constants_v2(concrete)

            conv = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
            conv.optimizations = []
            conv._experimental_lower_tensor_list_ops = True
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            print("[child] Trying: Frozen ConcreteFunction + BUILTINS (FP32)")
            tfl = conv.convert()
            save_bytes(tfl)
            return 0

        if mode == "F_freeze_fn_select_tf_ops":
            @tf.function(input_signature=[tf.TensorSpec([1,224,224,3], tf.float32, name="input")])
            def serving(x):
                y = model(x, training=False)
                y = tf.identity(y, name="output")
                return y

            concrete = serving.get_concrete_function()
            frozen = convert_variables_to_constants_v2(concrete)

            conv = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
            conv.optimizations = []
            conv._experimental_lower_tensor_list_ops = True
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            print("[child] Trying: Frozen ConcreteFunction + SELECT_TF_OPS (FP32)")
            tfl = conv.convert()
            save_bytes(tfl)
            return 0

        print(f"[child] Unknown mode: {mode}")
        return 3

    except Exception as e:
        print(f"[child] FAILED ({mode}): {e}")
        return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do", type=str, default=None)
    args = parser.parse_args()

    if args.do:
        rc = child_convert(args.do)
        sys.exit(rc)

    # Parent: try robust fallbacks first (freeze), then regular paths if you want.
    attempts = [
        "E_freeze_fn_builtins",
        "F_freeze_fn_select_tf_ops",
    ]

    py = sys.executable
    env = os.environ.copy()
    # Harmless: avoid oneDNN numerics during conversion
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    for m in attempts:
        print(f"[parent] Spawning attempt: {m}")
        rc = subprocess.call([py, __file__, "--do", m], env=env)
        if rc == 0:
            print(f"[parent] Success with {m}")
            print(f"[parent] TFLite at: {TFLITE_PATH}")
            sys.exit(0)
        else:
            print(f"[parent] Attempt {m} failed (rc={rc}), trying next...")

    sys.exit("All conversion strategies failed.")


if __name__ == "__main__":
    main()

