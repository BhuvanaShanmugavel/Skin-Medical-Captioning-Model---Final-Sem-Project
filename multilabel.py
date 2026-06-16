PHASE 1


# === Phase 1: Initial Training on 62 Labeled Images ===
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# === Load CSV Labels ===
label_csv_path = "/path/to/labels.csv"
data_dir = "/path/to/62_images/"
df = pd.read_csv(label_csv_path)

# === One-Hot Encode Keywords ===
mlb = MultiLabelBinarizer()
y_labels = mlb.fit_transform(df['keywords'].str.split(','))

# === Load and Preprocess 62 Images ===
X_imgs = []
for fname in df['filename']:
    img = cv2.imread(os.path.join(data_dir, fname))
    img = cv2.resize(img, (224, 224)) / 255.0
    X_imgs.append(img)
X_imgs = np.array(X_imgs)

# === Load Feature Extractors ===
discriminator = tf.keras.models.load_model("/path/to/discriminator_model.keras", compile=False)
autoencoder = tf.keras.models.load_model("/path/to/autoencoder_model.keras", compile=False)
disc_feat_extractor = Model(discriminator.input, discriminator.layers[-2].output)
ae_feat_extractor = Model(autoencoder.input, autoencoder.layers[-2].output)

# === Extract Features ===
disc_feats = disc_feat_extractor.predict(X_imgs)
ae_feats = ae_feat_extractor.predict(X_imgs)
X_combined = np.concatenate([disc_feats, ae_feats], axis=1)

# === Split & Train Classifier ===
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_labels, test_size=0.2, random_state=42)
inputs = Input(shape=X_train.shape[1:])
x = Dense(128, activation='relu')(inputs)
outputs = Dense(len(mlb.classes_), activation='sigmoid')(x)
classifier = Model(inputs, outputs)
classifier.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16)
classifier.save("/path/to/multilabel_classifier_model.keras")

# === Save train variables for later reuse ===
np.save("/path/to/X_train_feats.npy", X_combined)
np.save("/path/to/y_train_labels.npy", y_labels)


PHASE 2

# Phase 2: Train classifier on 7136 images (with pseudo-labels)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import os

# === Load pretrained models (paths to your Discriminator and Autoencoder) ===
discriminator = load_model("/path/to/discriminator_model.keras", compile=False)
autoencoder = load_model("/path/to/autoencoder_model.keras", compile=False)

# === Load and preprocess 7136 training images ===
X_train_all = np.load("/path/to/7136_images.npy")  # shape: (7136, 224, 224, 3)
X_train_all = X_train_all.astype(np.float32) / 255.0

# === Extract Features ===
discriminator_feature_model = Model(discriminator.input, discriminator.layers[-2].output)
autoencoder_feature_model = Model(autoencoder.input, autoencoder.layers[-2].output)

features_disc = discriminator_feature_model.predict(X_train_all, batch_size=32)
features_auto = autoencoder_feature_model.predict(X_train_all, batch_size=32)
features_combined = np.concatenate([features_disc, features_auto], axis=-1)

# === Load initial model trained on 62 samples ===
initial_classifier = load_model("/path/to/initial_multilabel_classifier.keras")

# === Continue training on pseudo-labeled data ===
# Load pseudo-labels (same shape as labels_62.npy from Phase 1)
pseudo_labels = np.load("/path/to/pseudo_labels_for_7136.npy")

history = initial_classifier.fit(
    features_combined, pseudo_labels,
    epochs=25,
    batch_size=32,
    validation_split=0.1
)

# Save updated model
initial_classifier.save("/path/to/final_multilabel_classifier.keras")
print("✅ Final classifier saved.")



PHASE 3

# Phase 3: Test classifier on 800 test images
# === Load and preprocess test images ===
X_test = np.load("/path/to/800_test_images.npy")
X_test = X_test.astype(np.float32) / 255.0

# === Extract test features ===
test_features_disc = discriminator_feature_model.predict(X_test, batch_size=32)
test_features_auto = autoencoder_feature_model.predict(X_test, batch_size=32)
test_features_combined = np.concatenate([test_features_disc, test_features_auto], axis=-1)

# === Load final classifier ===
final_model = load_model("/path/to/final_multilabel_classifier.keras")

# === Predict ===
predictions = final_model.predict(test_features_combined, batch_size=32)

# === Save predictions ===
np.save("/path/to/test_predictions.npy", predictions)
print("✅ Saved test predictions.")








DEEPSEEK CODE 

# === PHASE 2: Train classifier on 7136 images (with pseudo-labels) ===
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
import os

# === Paths (aligned with Phase 1) ===
BASE_DIR = "/SASTRA-NEW-CLUSTER/users/bhuvana/skin_dataset/"
discriminator_path = os.path.join(BASE_DIR, "unet_FinalDiscriminator.keras")
autoencoder_path = os.path.join(BASE_DIR, "finalmodel1ac.keras")
initial_classifier_path = os.path.join(BASE_DIR, "initial_multilabel_classifier.keras")

# === Load pretrained models ===
discriminator = load_model(discriminator_path, compile=False)
autoencoder = load_model(autoencoder_path, compile=False)

# === Load and preprocess 7136 training images ===
X_train_all = np.load(os.path.join(BASE_DIR, "7136_images.npy"))  # shape: (7136, 224, 224, 3)
X_train_all = X_train_all.astype(np.float32) / 255.0

# === Feature extraction (aligned with Phase 1) ===
disc_feat_model = Model(discriminator.input, discriminator.layers[-2].output)
ae_feat_model = Model(autoencoder.input, autoencoder.layers[-2].output)

disc_feats = disc_feat_model.predict(X_train_all, batch_size=32)
ae_feats = ae_feat_model.predict(X_train_all, batch_size=32)
X_combined = np.concatenate([disc_feats, ae_feats], axis=1)  # Note: axis=1 for consistency

# === Load initial model ===
classifier = load_model(initial_classifier_path)

# === Train with pseudo-labels ===
pseudo_labels = np.load(os.path.join(BASE_DIR, "pseudo_labels_for_7136.npy"))

history = classifier.fit(
    X_combined, pseudo_labels,
    epochs=25,
    batch_size=32,
    validation_split=0.1
)

# Save final model (same format as Phase 1)
final_classifier_path = os.path.join(BASE_DIR, "final_multilabel_classifier.keras")
classifier.save(final_classifier_path)
print("✅ Phase 2 complete: Final classifier saved at", final_classifier_path)

# === PHASE 3: Test classifier on 800 test images ===
# Load test data
X_test = np.load(os.path.join(BASE_DIR, "800_test_images.npy"))
X_test = X_test.astype(np.float32) / 255.0

# Extract test features (same as Phase 1/2)
test_disc_feats = disc_feat_model.predict(X_test, batch_size=32)
test_ae_feats = ae_feat_model.predict(X_test, batch_size=32)
test_combined = np.concatenate([test_disc_feats, test_ae_feats], axis=1)

# Load final model
final_model = load_model(final_classifier_path)

# Predict and save
predictions = final_model.predict(test_combined, batch_size=32)
np.save(os.path.join(BASE_DIR, "test_predictions.npy"), predictions)

print("\n📊 Phase 3 Results:")
print(f"- Predictions shape: {predictions.shape}")
print(f"- Sample prediction: {predictions[0]}")
print("✅ Test predictions saved.")