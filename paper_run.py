import os
import sys
import subprocess
import json
import glob
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.metrics import classification_report, f1_score

# ==========================================
# 1. Environment Setup & Dependencies
# ==========================================
print("Checking and installing dependencies...")
required_packages = ['kaggle', 'tensorflow', 'scikit-learn', 'matplotlib']
installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')

for pkg in required_packages:
    if pkg not in installed_packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

# ==========================================
# 2. Authentication & Data Download
# ==========================================
print("Configuring Kaggle credentials...")
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

creds = {"username":"ronrexy","key":"35697bceee2dff0e1af9b46fd926f4b5"}
with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
    json.dump(creds, f)
os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

RAW_DATA_DIR = "oasis_raw"
if not os.path.exists(RAW_DATA_DIR):
    print("Downloading raw dataset...")
    subprocess.check_call(['kaggle', 'datasets', 'download', '-d', 'ninadaithal/imagesoasis', '--unzip', '-p', RAW_DATA_DIR])

# Locate source directory
found = glob.glob(os.path.join(RAW_DATA_DIR, "**", "Mild Dementia"), recursive=True)
SOURCE_PATH = os.path.dirname(found[0])

# ==========================================
# 3. Data Processing: Subject-Independent Split
# ==========================================
BASE_DIR = "processed_data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
CLASSES = sorted(["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"])

def organize_data_by_patient():
    """
    Sorts images based on Patient ID to ensure subject independence
    between training and validation sets.
    """
    if os.path.exists(BASE_DIR):
        print("Processed data directory already exists. Skipping sorting.")
        return

    print("Sorting data by Patient ID to prevent data leakage...")
    
    # Map Patient IDs to their respective files
    patient_map = {}
    
    for class_name in CLASSES:
        class_path = os.path.join(SOURCE_PATH, class_name)
        files = os.listdir(class_path)
        
        for f in files:
            # Filename extraction: OAS1_0031_MR1... -> Patient ID: OAS1_0031
            parts = f.split('_')
            if len(parts) > 2:
                patient_id = f"{parts[0]}_{parts[1]}"
                
                if patient_id not in patient_map:
                    patient_map[patient_id] = {'class': class_name, 'files': []}
                
                patient_map[patient_id]['files'].append(os.path.join(class_path, f))

    patient_ids = list(patient_map.keys())
    random.seed(42)
    random.shuffle(patient_ids)
    
    # Split Patients (80% Train, 20% Validation)
    split_idx = int(0.8 * len(patient_ids))
    train_patients = patient_ids[:split_idx]
    val_patients = patient_ids[split_idx:]
    
    print(f"Total Patients: {len(patient_ids)}")
    print(f"Training Patients: {len(train_patients)}")
    print(f"Validation Patients: {len(val_patients)}")
    
    # Create directory structure and symlinks
    for p_list, target_dir in [(train_patients, TRAIN_DIR), (val_patients, VAL_DIR)]:
        for p_id in p_list:
            data = patient_map[p_id]
            class_name = data['class']
            dest_dir = os.path.join(target_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for src_file in data['files']:
                file_name = os.path.basename(src_file)
                os.symlink(os.path.abspath(src_file), os.path.join(dest_dir, file_name))
                
    print("Data sorting complete.")

organize_data_by_patient()

# ==========================================
# 4. Hardware Configuration
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Enable Mixed Precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()

# ==========================================
# 5. Training Configuration
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 128
EPOCHS = 20

def get_class_weights():
    counts = {}
    for c in CLASSES:
        counts[c] = len(os.listdir(os.path.join(TRAIN_DIR, c)))
    total = sum(counts.values())
    return {i: total / (4 * c) if c > 0 else 0 for i, c in enumerate(counts.values())}

class_weights = get_class_weights()
print(f"Class Weights: {class_weights}")

# ==========================================
# 6. Model & Pipeline Definition
# ==========================================
class MultiClassFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

def build_dataset(model_type, directory):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory, seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    
    def preprocess(x, y):
        x = tf.cast(x, tf.float32)
        if model_type == 'resnet':
            x = resnet_preprocess(x)
        else:
            x = mobilenet_preprocess(x)
        return x, y
        
    return ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 7. Training Loop
# ==========================================
results = {}

for m_name in ['resnet', 'mobilenet']:
    print(f"\nStarting training for model: {m_name.upper()}")
    tf.keras.backend.clear_session()
    
    train_ds = build_dataset(m_name, TRAIN_DIR)
    val_ds = build_dataset(m_name, VAL_DIR)
    
    with strategy.scope():
        if m_name == 'resnet':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        else:
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        
        base_model.trainable = True
        
        inputs = tf.keras.Input(shape=(224,224,3))
        x = base_model(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(4, activation='softmax', dtype='float32')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Using a lower learning rate for fine-tuning
        model.compile(optimizer=optimizers.Adam(5e-5), 
                      loss=MultiClassFocalLoss(), 
                      metrics=['accuracy'])

    print(f"Fitting {m_name}...")
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ],
        verbose=1
    )
    
    print(f"Evaluating {m_name}...")
    y_true, y_pred = [], []
    for x, y in val_ds:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))
        
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    f1 = f1_score(y_true, y_pred, average='weighted')
    results[m_name] = {'f1': f1, 'report': report}
    
    model.save(f"{m_name}_patient_split.keras")
    print(report)

# ==========================================
# 8. Saving Results
# ==========================================
with open('paper_results.txt', 'w') as f:
    f.write("Subject-Independent Split Results\n")
    f.write("=================================\n")
    for m, res in results.items():
        f.write(f"\n--- {m.upper()} ---\n")
        f.write(f"Weighted F1: {res['f1']}\n")
        f.write(res['report'] + "\n")

print("\nExecution complete.")