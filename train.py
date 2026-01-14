import os
import sys
import subprocess
import json
import glob
import time

# ==========================================
# 1. AUTO-INSTALL DEPENDENCIES
# ==========================================
print(" Setting up Environment...")
required = ['kaggle', 'tensorflow', 'scikit-learn', 'matplotlib', 'pandas']
installed = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
for pkg in required:
    if pkg not in installed:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.metrics import classification_report, f1_score

# ==========================================
# 2. KAGGLE AUTHENTICATION
# ==========================================
print(" configuring Kaggle Credentials...")
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)


creds = {"username":"ronrexy","key":"35697bceee2dff0e1af9b46fd926f4b5"}

with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
    json.dump(creds, f)
os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

# 3. DATASET DOWNLOAD & SETUP

DATA_ROOT = "oasis_data"
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT, exist_ok=True)
    print(" Downloading OASIS Dataset...")
    subprocess.check_call(['kaggle', 'datasets', 'download', '-d', 'ninadaithal/imagesoasis', '--unzip', '-p', DATA_ROOT])
else:
    print(" Dataset already present.")

# Find the actual data directory (handling unzip structure)
found = glob.glob(os.path.join(DATA_ROOT, "**", "Mild Dementia"), recursive=True)
if not found: 
    raise FileNotFoundError("Could not locate dataset structure after download.")
DATASET_PATH = os.path.dirname(found[0])
print(f" Data Source: {DATASET_PATH}")


# 4. L40S HARDWARE CONFIG

print("🔧 Configuring L40S GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" Active GPU: {len(gpus)}x NVIDIA L40S Detected")
    except RuntimeError as e: print(e)

# Enable Mixed Precision (Crucial for L40S Speed)
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


# 5. HYPERPARAMETERS

IMG_SIZE = (224, 224)
# L40S (48GB VRAM) allows large batches = faster training
BATCH_SIZE = 128 
EPOCHS = 20
CLASSES = sorted(["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"])


# 6. PIPELINE UTILITIES

def get_class_weights():
    counts = {}
    for c in CLASSES:
        path = os.path.join(DATASET_PATH, c)
        counts[c] = len(os.listdir(path)) if os.path.exists(path) else 0
    total = sum(counts.values())
    weights = {i: total / (4 * c) if c > 0 else 0 for i, c in enumerate(counts.values())}
    print(f"⚖️ Class Weights: {weights}")
    return weights

class_weights = get_class_weights()

class MultiClassFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Focal Loss logic
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

def build_ds(model_type, subset):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, validation_split=0.2, subset=subset, seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    
    def preprocess(x, y):
        x = tf.cast(x, tf.float32)
        if model_type == 'resnet':
            x = resnet_preprocess(x)
        else:
            x = mobilenet_preprocess(x)
        return x, y
        
    # L40S Optimization: Parallel mapping + Prefetching
    return ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# 7. MAIN TRAINING LOOP

strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()

results = {}

for m_name in ['resnet', 'mobilenet']:
    print(f"\n{'='*40}")
    print(f"⚡ STARTING: {m_name.upper()}")
    print(f"{'='*40}")
    
    # Clean memory
    tf.keras.backend.clear_session()
    
    train_ds = build_ds(m_name, "training")
    val_ds = build_ds(m_name, "validation")
    
    with strategy.scope():
        # Load Base Model
        if m_name == 'resnet':
            base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        else:
            base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        
        base.trainable = True # Fine-tuning
        
        inputs = tf.keras.Input(shape=(224,224,3))
        x = base(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(4, activation='softmax', dtype='float32')(x)
        
        model = tf.keras.Model(inputs, out)
        model.compile(optimizer=optimizers.Adam(1e-4), 
                      loss=MultiClassFocalLoss(), 
                      metrics=['accuracy'])

    # Training
    print(f"Training {m_name}...")
    start = time.time()
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        ],
        verbose=1
    )
    print(f" Training Time: {time.time() - start:.2f}s")
    
    # Evaluation
    print(f" Evaluating {m_name}...")
    y_true, y_pred = [], []
    for x, y in val_ds:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))
        
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    f1 = f1_score(y_true, y_pred, average='weighted')
    results[m_name] = {'f1': f1, 'report': report}
    
    # Save Artifacts
    model.save(f"{m_name}_oasis_l40s.keras")
    print(report)


# 8. FINAL OUTPUT

print("\n" + "="*40)
print(" FINAL COMPARISON (L40S)")
print("="*40)
for m, res in results.items():
    print(f"{m.upper():<15} | Weighted F1: {res['f1']:.4f}")

# Write to file
with open('comparison_results.txt', 'w') as f:
    for m, res in results.items():
        f.write(f"--- {m.upper()} ---\n")
        f.write(res['report'] + "\n")

print("\n Script Complete. Don't forget to STOP the Studio!")
