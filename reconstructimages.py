import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# === CONFIGURATION === #
model_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/bestmodel.keras"
data_path = "/nfsshare/bhuvana/Dataset/dataset/Dermnet_TrainTestSplit/PreprocessedImages/X_train.npy"
save_dir = "./reconstructions"
num_images_to_save = 10

# === CUSTOM LOSS FUNCTION (REQUIRED FOR LOADING MODEL) === #
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.7 * mse + 0.3 * mae

# === LOAD MODEL === #
model = tf.keras.models.load_model(model_path, custom_objects={"combined_loss": combined_loss})

# === MEMORY-EFFICIENT LOAD === #
X_memmap = np.load(data_path, mmap_mode='r')
total_samples = X_memmap.shape[0]
sample_indices = np.random.choice(total_samples, num_images_to_save, replace=False)
X_samples = X_memmap[sample_indices].astype('float32') / 255.0

# === PREDICT (RECONSTRUCT) === #
reconstructed = model.predict(X_samples, batch_size=1)

# === SAVE SIDE-BY-SIDE IMAGES === #
os.makedirs(save_dir, exist_ok=True)

for i in range(num_images_to_save):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    axes[0].imshow(X_samples[i])
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(reconstructed[i])
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"reconstruction_{i+1}.png")
    plt.savefig(save_path)
    plt.close()

print(f"\n? {num_images_to_save} reconstructed image comparisons saved to: {save_dir}")
