import os
# Suppress noisy TensorFlow C++ / oneDNN informational messages in the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Keras components referenced directly from tf.keras (eliminates VS Code / Pylance import squiggles)
MobileNetV2 = tf.keras.applications.MobileNetV2
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
import splitfolders

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SPLIT_DIR = PROJECT_ROOT / 'data' / 'split'
MODEL_DIR = PROJECT_ROOT / 'model'
REPORT_DIR = PROJECT_ROOT / 'report'

# Resolve the original dataset path only if the existing split is not available.
possible_paths = [
    r'C:\Users\amitr\Downloads\archive (4)\plantvillage dataset\color',
    r'C:\Users\amitr\Downloads\plantvillage dataset\color',
]
dataset_path = next((p for p in possible_paths if os.path.exists(p)), None)

if not (DATA_SPLIT_DIR / 'train').exists() and not dataset_path:
    raise FileNotFoundError("Could not find the dataset folder in Downloads. Please verify the path.")

if dataset_path:
    print(f"Using dataset from: {dataset_path}")

# Step 1: Split into 70/15/15 — do this ONCE before training
if not (DATA_SPLIT_DIR / 'train').exists():
    print("Splitting dataset into train/val/test (70/15/15)...")
    splitfolders.ratio(str(dataset_path), output=str(DATA_SPLIT_DIR), seed=42, ratio=(0.70, 0.15, 0.15))
else:
    print("data/split already exists. Skipping split step.")

# Step 2: Load the split data
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.1,
    height_shift_range=0.1,
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    str(DATA_SPLIT_DIR / 'train'),
    target_size=(224, 224),
    batch_size=32,
    seed=42,
)
class_names = [name for name, index in sorted(train_data.class_indices.items(), key=lambda item: item[1])]

val_data = eval_datagen.flow_from_directory(
    str(DATA_SPLIT_DIR / 'val'),
    target_size=(224, 224),
    batch_size=32,
    classes=class_names,
    shuffle=False,
)
test_data = eval_datagen.flow_from_directory(
    str(DATA_SPLIT_DIR / 'test'),
    target_size=(224, 224),
    batch_size=32,
    classes=class_names,
    shuffle=False,
)

# Save class mapping so inference app can map prediction indices to disease names
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
with open(MODEL_DIR / 'class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f, indent=4)
print(f"Saved {train_data.num_classes} classes to model/class_indices.json")

# Load MobileNetV2, freeze its weights (we only train our new top layer)
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

# Add our classification head on top
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(train_data.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

best_model_path = MODEL_DIR / 'best_model.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(best_model_path),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1,
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=2,
    restore_best_weights=True,
    verbose=1,
)

# First train only the new classification head.
model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    callbacks=[checkpoint, early_stopping],
)

# Fine-tune only the final 20 MobileNetV2 layers with a small learning rate.
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
for layer in base.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[checkpoint, early_stopping],
)

if best_model_path.exists():
    model = models.load_model(best_model_path)

# Evaluate once on the untouched test set.
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
test_data.reset()
probabilities = model.predict(test_data, verbose=1)
predicted_labels = probabilities.argmax(axis=1)
true_labels = test_data.classes
macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
)
matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(class_names))))

with open(REPORT_DIR / 'classification_report.txt', 'w') as f:
    f.write(f'Test accuracy: {test_accuracy:.4f}\n')
    f.write(f'Macro-F1: {macro_f1:.4f}\n\n')
    f.write(report)
with open(REPORT_DIR / 'evaluation_metrics.json', 'w') as f:
    json.dump(
        {'test_loss': float(test_loss), 'test_accuracy': float(test_accuracy), 'macro_f1': float(macro_f1)},
        f,
        indent=4,
    )

plt.figure(figsize=(16, 14))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('AgriSmart AI Test Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORT_DIR / 'confusion_matrix.png', dpi=150)
plt.close()

# Save the final best model in the required format.
model.save(MODEL_DIR / 'agrismart_model.h5')
model.save(MODEL_DIR / 'agrismart_model.keras')
print(f'Test accuracy: {test_accuracy:.4f}')
print(f'Macro-F1: {macro_f1:.4f}')
print('Training and evaluation complete. Model saved to model/agrismart_model.keras')