

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === PATHS === #
model_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/finalmodel.keras"
test_data_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/X_test.npy"

# === CUSTOM LOSS === #
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.7 * mse + 0.3 * mae

# === LOAD MODEL === #
model = tf.keras.models.load_model(model_path, custom_objects={'combined_loss': combined_loss})

# === LOAD TEST DATA === #
X_test = np.load(test_data_path).astype('float32') / 255.0

# === PREDICT === #
reconstructed = model.predict(X_test, batch_size=32)

# === METRICS === #
mse = mean_squared_error(X_test.flatten(), reconstructed.flatten())
mae = mean_absolute_error(X_test.flatten(), reconstructed.flatten())
print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
