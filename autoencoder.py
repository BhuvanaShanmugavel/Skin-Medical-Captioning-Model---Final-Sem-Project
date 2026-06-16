import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === CONFIGURATION === #
train_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/X_train.npy"
test_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/X_test.npy"
final_model_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/finalmodel.keras"
best_model_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/bestmodel.keras"

batch_size = 32
epochs = 50
val_split = 0.15
image_shape = (224, 224, 3)

# === LOAD TEST DATA === #
X_test = np.load(test_path).astype('float32') / 255.0

# === TRAIN DATA SHAPE === #
total_train_samples = 7136
val_size = int(total_train_samples * val_split)
train_size = total_train_samples - val_size

# === AUTOENCODER === #
def build_autoencoder(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=2, padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same', kernel_regularizer=l2(0.001))(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), strides=2, padding='same', kernel_regularizer=l2(0.001))(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)
    x = ReLU()(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = ReLU()(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = ReLU()(x)

    outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(inputs, outputs)

# === ACCURACY METRIC === #
def accuracy_metric(y_true, y_pred, threshold=0.05):
    diff = np.abs(y_true - y_pred)
    correct = np.sum(diff < threshold)
    total = np.prod(y_true.shape)
    return correct / total

# === DATA GENERATOR === #
def data_generator(path, indices, batch_size):
    data = np.load(path, mmap_mode='r')
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = data[batch_idx].astype('float32') / 255.0
            yield batch, batch

# === TRAIN/VAL SPLIT === #
all_indices = np.arange(total_train_samples)
train_indices, val_indices = train_test_split(all_indices, test_size=val_split, random_state=42)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_path, train_indices, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32))
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_path, val_indices, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32))
).prefetch(tf.data.AUTOTUNE)

# === BUILD AND COMPILE MODEL === #
autoencoder = build_autoencoder()
print("\n?? Model Summary:")
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === TRAINING LOOP WITH TQDM === #
steps_per_epoch = len(train_indices) // batch_size
val_steps = len(val_indices) // batch_size
best_val_loss = float('inf')

print("\n?? Starting Training...\n")

for epoch in range(epochs):
    print(f"\n?? Epoch {epoch + 1}/{epochs}")
    progbar = tqdm(range(steps_per_epoch), desc="Training", unit="batch")

    # Training loop
    train_loss = []
    train_mae = []
    train_iter = iter(train_dataset)
    for _ in progbar:
        x_batch, y_batch = next(train_iter)
        metrics = autoencoder.train_on_batch(x_batch, y_batch)
        train_loss.append(metrics[0])
        train_mae.append(metrics[1])
        progbar.set_postfix({"Loss": f"{np.mean(train_loss):.6f}", "MAE": f"{np.mean(train_mae):.6f}"})

    # Validation loop
    val_loss = []
    val_mae = []
    val_iter = iter(val_dataset)
    for _ in range(val_steps):
        x_val, y_val = next(val_iter)
        metrics = autoencoder.test_on_batch(x_val, y_val)
        val_loss.append(metrics[0])
        val_mae.append(metrics[1])

    avg_val_loss = np.mean(val_loss)
    avg_val_mae = np.mean(val_mae)
    print(f"? Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        autoencoder.save(best_model_path)
        print(f"?? Best model saved at epoch {epoch + 1}")

# === SAVE FINAL MODEL === #
autoencoder.save(final_model_path)
print(f"\n? Final model saved to: {final_model_path}")

# === EVALUATE ON TEST SET === #
print("\n?? Evaluating on Test Data...")
test_loss, test_mae = autoencoder.evaluate(X_test, X_test, verbose=1)
reconstructed = autoencoder.predict(X_test, verbose=1)
accuracy = accuracy_metric(X_test, reconstructed)

print(f"?? Test MSE: {test_loss:.6f}")
print(f"?? Test MAE: {test_mae:.6f}")
print(f"?? Pixel Accuracy (<0.05): {accuracy:.4f}")
