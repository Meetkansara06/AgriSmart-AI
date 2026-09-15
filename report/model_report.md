# AgriSmart AI — Model Report

**Team:** L.J. Institute of Engineering & Technology [C-433]  
**Hackathon:** SIH 2026 (Internal)

---

## Problem Statement

Detect crop leaf diseases from images and provide actionable farming advice. The system must classify leaf images into disease categories with high accuracy and deliver farmer-friendly precautions, irrigation guidance, sustainability scoring, and crop recommendations.

---

## Models Used

### 1. Disease Detection — MobileNetV2 (Transfer Learning)

| Property | Detail |
|---|---|
| Base Model | MobileNetV2 (pre-trained on ImageNet) |
| Framework | TensorFlow 2.21.0 + Keras |
| Input Size | 224 × 224 × 3 (RGB) |
| Output | 18 classes (softmax) |
| Total Parameters | ~2.3M (MobileNetV2) + 18 dense |
| Training Data | PlantVillage (~19,600 images, 70/15/15 split) |

**Architecture:**

```
MobileNetV2 (ImageNet weights) → GlobalAveragePooling2D → Dense(18, softmax)
```

**Training Strategy:**

| Phase | What | Optimizer | LR | Epochs |
|---|---|---|---|---|
| Phase 1 | Train dense head only (base frozen) | Adam | Default | 3 (early stop) |
| Phase 2 | Fine-tune last 20 base layers + head | Adam | 1e-5 | 10 (early stop) |

**Data Augmentation (training only):** Horizontal flip, rotation ±20°, zoom 20%, brightness [0.7–1.3], width/height shift 10%.

**Preprocessing:** `mobilenet_v2.preprocess_input` — scales pixel values to [-1, 1].

**Callbacks:** ModelCheckpoint (best `val_accuracy`) + EarlyStopping (patience=2, restores best weights).

### 2. Crop Recommendation — Random Forest

| Property | Detail |
|---|---|
| Algorithm | RandomForestClassifier (scikit-learn) |
| n_estimators | 100 |
| random_state | 42 |
| Features | N, P, K, temperature, humidity, ph, rainfall |
| Output | 22 crop classes |
| Training Data | Kaggle Crop Recommendation (2,200 rows, 80/20 stratified split) |

---

## Results

### Disease Detection (MobileNetV2)

Evaluated on held-out PlantVillage test split (2,950 images, never seen during training).

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.63%** |
| **Macro F1-Score** | **97.19%** |
| Test Loss | 0.0707 |

**Per-Class F1-Scores:**

| Class | F1 | Support | | Class | F1 | Support |
|---|---|---|---|---|---|---|
| Apple — Apple Scab | 0.98 | 95 | | Pepper Bell — Healthy | 0.99 | 223 |
| Apple — Black Rot | 0.99 | 94 | | Potato — Early Blight | 0.99 | 150 |
| Apple — Healthy | 0.99 | 248 | | Potato — Late Blight | 0.96 | 150 |
| Corn — Gray Leaf Spot | 0.99 | 78 | | Potato — Healthy | 0.90 | 24 |
| Corn — Common Rust | 0.99 | 180 | | Tomato — Bacterial Spot | 0.99 | 320 |
| Corn — Healthy | 1.00 | 175 | | Tomato — Early Blight | 0.90 | 150 |
| Grape — Black Rot | 1.00 | 177 | | Tomato — Late Blight | 0.95 | 287 |
| Grape — Healthy | 0.97 | 64 | | Tomato — Leaf Mold | 0.94 | 144 |
| Pepper Bell — Bacterial Spot | 0.98 | 151 | | Tomato — Healthy | 0.99 | 240 |

**Weakest classes:** Potato Healthy (F1=0.90, only 24 test samples), Tomato Early Blight (F1=0.90).

### Crop Recommendation (Random Forest)

| Metric | Value |
|---|---|
| **Test Accuracy** | **99.55%** |
| **Macro F1-Score** | **0.9955** |
| Classes | 22 crops |

---

## Bonus Module Formulas

### Sustainability Score (0–100)

```
score = (water_score × 0.40) + (health_score × 0.40) + (resource_score × 0.20)

water_score    = max(0, 1 − water_used / max_water) × 100
health_score   = 0 if disease_detected else 100
resource_score = 100 − fertilizer_level
```

| Badge | Threshold |
|---|---|
| Platinum | score > 85 |
| Green Guardian | score > 70 |
| Needs Improvement | score 50–70 |
| At Risk | score < 50 |

### Irrigation Rules (priority order)

| Condition | Decision |
|---|---|
| rain > 10 mm | Delay irrigation |
| rain > 4 mm + sandy soil | Light irrigation |
| rain > 4 mm + other soil | Skip irrigation |
| rain ≤ 4 mm + seedling | Irrigate gently |
| rain ≤ 4 mm + other stage | Irrigate today |

### Weather Intelligence

- **Source:** Open-Meteo API (free, no API key)
- **Data:** Tomorrow's precipitation (mm), max/min temperature (°C)
- **Feeds into:** Irrigation decision module

---

## Limitations

| Limitation | Impact |
|---|---|
| PlantVillage is studio-quality | Field-condition images may yield lower accuracy |
| 18-class scope only | Apple, Corn, Grape, Pepper, Potato, Tomato — no other crops |
| Small Potato Healthy support (24) | Lowest F1 score in the model |
| CPU-only on Windows | TensorFlow GPU not supported natively; inference takes ~8s |
| Weather module needs internet | Offline mode not available for irrigation advice |

---

## Reproducibility

```powershell
git clone <repo_url>
cd AgriSmart-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe model\predict.py --image "path\to\leaf.jpg"
```

Expected: disease class + confidence printed in < 10 seconds after model load.
