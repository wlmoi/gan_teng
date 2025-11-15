import tensorflow as tf
import keras
import tf2onnx

# === CONFIG ===
keras_model_name = "mnist_model.keras"
onnx_model_name  = "mnist_model.onnx"

# === LOAD MODEL ===
print("Loading Keras model...")
model = keras.saving.load_model(keras_model_name)

# === CONVERT TO ONNX ===
print("Converting to ONNX...")
spec = (tf.TensorSpec((None, 784), tf.float32, name="input"),)

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13
)

# === SAVE FILE ===
with open(onnx_model_name, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"[DONE] Converted → {onnx_model_name}")


'''
cd D:\GANTENG
conda create -y -n tf310 python=3.10
conda activate tf310
python -m pip install --upgrade pip
python -m pip install tensorflow keras tf2onnx onnx
python convert_keras_to_onnx.py

'''