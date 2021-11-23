from tensorflow.keras.models import load_model

model = load_model("/build/mnist.h5")
model.summary()
