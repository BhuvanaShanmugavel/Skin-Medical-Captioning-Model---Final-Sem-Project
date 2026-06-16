import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, ReLU, Dropout,
                                     BatchNormalization, GaussianNoise)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === CONFIGURATION === #
train_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/X_train.npy"
final_model_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/finalmodel1.h5"
best_model_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/bestmodel1.h5"

batch_size = 32
epochs = 5
val_split = 0.15
image_shape = (224, 224, 3)
target_mse_range = (0.02, 0.06)
target_mae_range = (0.11, 0.15)

# === MEMORY-EFFICIENT LOADING === #
X_memmap = np.load(train_path, mmap_mode='r')
total_samples = X_memmap.shape[0]
indices = np.arange(total_samples)
train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)

def create_dataset(memmap_data, indices, batch_size):
    def generator():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = memmap_data[batch_indices].astype('float32') / 255.0
            yield batch, batch
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *image_shape), dtype=tf.float32)
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(X_memmap, train_idx, batch_size)
val_dataset = create_dataset(X_memmap, val_idx, batch_size)

# === BUILD AUTOENCODER === #
def build_autoencoder(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.05)(inputs)

    x = Conv2D(64, (3, 3), strides=2, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), strides=2, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(inputs, outputs)

# === COMBINED LOSS === #
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.7 * mse + 0.3 * mae

# === WARM-UP + DECAY LR === #
class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        decayed_lr = self.initial_lr * tf.pow(self.decay_rate, (step - self.warmup_steps) / self.decay_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

lr_schedule = WarmupExponentialDecay(1e-3, warmup_steps=500, decay_steps=1500, decay_rate=0.9)

# === COMPILE MODEL === #
autoencoder = build_autoencoder()
autoencoder.summary()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=combined_loss,
                    metrics=['mse', 'mae'])

# === TRAIN LOOP === #
print("\n?? Starting Training...\n")
best_saved = False

for epoch in range(epochs):
    print(f"\n?? Epoch {epoch + 1}/{epochs}")
    progbar = tqdm(enumerate(train_dataset), total=len(train_idx)//batch_size, desc="Training", unit="batch")
    train_loss, train_mse, train_mae = [], [], []

    for _, (x_batch, _) in progbar:
        metrics = autoencoder.train_on_batch(x_batch, x_batch)
        train_loss.append(metrics[0])
        train_mse.append(metrics[1])
        train_mae.append(metrics[2])
        progbar.set_postfix({
            "Loss": f"{np.mean(train_loss):.6f}",
            "MSE": f"{np.mean(train_mse):.6f}",
            "MAE": f"{np.mean(train_mae):.6f}"
        })

    # === VALIDATION === #
    val_loss, val_mse, val_mae = [], [], []
    for x_val, _ in val_dataset:
        metrics = autoencoder.test_on_batch(x_val, x_val)
        val_loss.append(metrics[0])
        val_mse.append(metrics[1])
        val_mae.append(metrics[2])

    avg_val_loss = np.mean(val_loss)
    avg_val_mse = np.mean(val_mse)
    avg_val_mae = np.mean(val_mae)
    print(f"?? Val Loss: {avg_val_loss:.6f}, MSE: {avg_val_mse:.6f}, MAE: {avg_val_mae:.6f}")

    if not best_saved and target_mse_range[0] <= avg_val_mse <= target_mse_range[1] and \
       target_mae_range[0] <= avg_val_mae <= target_mae_range[1]:
        autoencoder.save(best_model_path)
        print(f"? Best model saved at epoch {epoch + 1}")
        best_saved = True

# === FINAL SAVE === #
autoencoder.save(final_model_path)
print(f"\n? Final model saved to: {final_model_path}")
