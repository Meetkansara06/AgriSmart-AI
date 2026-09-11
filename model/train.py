import os
# Suppress noisy TensorFlow C++ / oneDNN informational messages in the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import tensorflow as tf

# Keras components referenced directly from tf.keras (eliminates VS Code / Pylance import squiggles)
MobileNetV2 = tf.keras.applications.MobileNetV2
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
import splitfolders

# Resolve local dataset path
possible_paths = [
    r'C:\Users\amitr\Downloads\archive (4)\plantvillage dataset\color',
    r'C:\Users\amitr\Downloads\plantvillage dataset\color',
]
dataset_path = next((p for p in possible_paths if os.path.exists(p)), None)

if not dataset_path:
    raise FileNotFoundError("Could not find the dataset folder in Downloads. Please verify the path.")

print(f"Using dataset from: {dataset_path}")

# Step 1: Split into 70/15/15 — do this ONCE before training
if not os.path.exists('data/split/train'):
    print("Splitting dataset into train/val/test (70/15/15)...")
    splitfolders.ratio(dataset_path, output='data/split', seed=42, ratio=(0.70, 0.15, 0.15))
else:
    print("data/split already exists. Skipping split step.")

# Step 2: Load the split data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'data/split/train',
    target_size=(224, 224),
    batch_size=32
)
val_data = val_datagen.flow_from_directory(
    'data/split/val',
    target_size=(224, 224),
    batch_size=32
)

# Save class mapping so inference app can map prediction indices to disease names
os.makedirs('model', exist_ok=True)
with open('model/class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f, indent=4)
print(f"Saved {train_data.num_classes} classes to model/class_indices.json")

# data/split/test/ is kept aside — never used during training, only for final metric reporting
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

# Train for just 3 epochs today — just proving it works
model.fit(train_data, validation_data=val_data, epochs=3)

# Save the model (both .h5 and .keras for maximum compatibility)
model.save('model/agrismart_model.h5')
try:
    model.save('model/agrismart_model.keras')
except Exception as e:
    print(f"Note: Could not save .keras: {e}")
print("Training complete! Model saved to model/agrismart_model.h5")