# AgriSmart AI 🌱

AI-powered crop disease detection and smart agriculture advisory system for sustainable farming.

---

## 1. Modules Built

### Core — Crop Disease Detection (`model/`)

The Core ML pipeline uses TensorFlow 2.21.0 and MobileNetV2 pretrained on ImageNet. It includes:

- Training-only image augmentation
- Official MobileNetV2 preprocessing
- Frozen-head training followed by fine-tuning of the final 20 base-model layers
- Validation-based checkpointing and early stopping
- Accuracy, Macro-F1, classification report, and confusion matrix evaluation
- A standalone image prediction interface

| File | Purpose |
|---|---|
| `model/train.py` | Transfer learning pipeline (MobileNetV2 on PlantVillage) |
| `model/predict.py` | Single-image inference — returns class label + confidence |
| `model/evaluate.py` | Batch evaluation on an external / held-out dataset |
| `model/class_indices.json` | Mapping of 18 class names to model output indices |

### Advisory & Bonus Modules (`src/`)

| File | Module | What it does |
|---|---|---|
| `src/precautions.py` | Disease Precautions | Farmer-friendly precaution steps for each of the 18 disease classes |
| `src/weather.py` | Weather Intelligence (Bonus C) | Calls Open-Meteo API (no key needed) — tomorrow's rainfall & temperature |
| `src/irrigation.py` | Smart Irrigation (Bonus B) | Rule-based decision combining weather forecast + crop stage + soil type |
| `src/sustainability.py` | Sustainability Score (Bonus D) | Formula-based 0–100 score from water efficiency, crop health, resource use |
| `src/crop_recommendation.py` | Crop Recommendation (Bonus A) | Recommends best crop based on soil N/P/K, pH, rainfall, temperature |

### Crop Recommendation

The Crop Recommendation module predicts a suitable crop from seven soil and
weather measurements: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and
`rainfall`. It uses the public Kaggle Crop Recommendation dataset from
`atharvaingle/crop-recommendation-dataset`, stored locally at
`data/crop_recommendation/Crop_recommendation.csv`.

The model is a `RandomForestClassifier` with `n_estimators=100` and
`random_state=42`. Training uses an 80/20 stratified train/test split. The
trained artifact is saved as `model/crop_rec_model.pkl`.

Measured results from the current training run:

- Dataset shape: `(2200, 8)`
- Crop classes: `22`
- Test accuracy: `99.55%`
- Macro-F1: `0.9955`

### Streamlit App (`app.py`)

Unified web interface wiring all modules into one farmer-facing product:
- Upload a leaf image → instant disease diagnosis + confidence
- Precaution & treatment plan per disease class
- Live weather forecast for entered coordinates
- Smart irrigation recommendation
- Sustainability score with eco-badge
- Crop recommendation from soil/climate inputs

---

## 2. Setup & Run Instructions

### Prerequisites
- Python 3.10 or higher
- Windows (instructions below use PowerShell)

### Create & activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the Streamlit App

```powershell
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### Train the disease detection model (optional — pre-trained model included)

```powershell
.\.venv\Scripts\python.exe model\train.py
```

The script reuses `data/split` when it already exists. It does not recreate or overwrite the existing split. Training outputs are saved to `model/` and `report/`.

### Predict One Image

```powershell
.\.venv\Scripts\python.exe model\predict.py --image "path\to\leaf.jpg"
```

The command prints the predicted disease class and confidence.

### Evaluate an Unseen Dataset

The external evaluator expects one subdirectory per class and never trains on the supplied data:

```powershell
.\.venv\Scripts\python.exe model\evaluate.py --data-dir "path\to\external_dataset"
```

Reports are written to `report/external_evaluation/` by default. Use the organizer or field-condition dataset here only for final evaluation. Do not add it to the training split.

### Train the Crop Recommendation model

```powershell
.\.venv\Scripts\python.exe model\train_crop.py --data "data\crop_recommendation\Crop_recommendation.csv"
```

### Run a Crop Recommendation prediction (CLI)

```powershell
.\.venv\Scripts\python.exe model\crop_predict.py --N 90 --P 42 --K 43 --temperature 25 --humidity 80 --ph 6.5 --rainfall 200
```

The Streamlit Crop Recommendation tab accepts the same seven inputs and calls
the saved model independently of the disease image workflow. The result is a
data-driven recommendation, not a guarantee of field performance; soil tests,
local agronomy, weather, and farmer expertise remain important.

---

## 3. Dataset Used

### Disease Detection — PlantVillage

| Property | Detail |
|---|---|
| Source | PlantVillage (~54k images, filtered to 18 classes) |
| Split Ratio | 70% train / 15% validation / 15% test (seed=42) |
| Split Tool | `split-folders` v0.6.1 |
| Image Size | Resized to 224 × 224 pixels at load time |

Split locations:

- `data/split/train`
- `data/split/val`
- `data/split/test`

The organizer or field-condition test dataset is reserved for measuring real-world generalization and is not used during training or model selection.

### 18 Supported Disease Classes

| # | Class Label | # | Class Label |
|---|---|---|---|
| 0 | Apple — Apple Scab | 9 | Pepper Bell — Healthy |
| 1 | Apple — Black Rot | 10 | Potato — Early Blight |
| 2 | Apple — Healthy | 11 | Potato — Late Blight |
| 3 | Corn — Gray Leaf Spot | 12 | Potato — Healthy |
| 4 | Corn — Common Rust | 13 | Tomato — Bacterial Spot |
| 5 | Corn — Healthy | 14 | Tomato — Early Blight |
| 6 | Grape — Black Rot | 15 | Tomato — Late Blight |
| 7 | Grape — Healthy | 16 | Tomato — Leaf Mold |
| 8 | Pepper Bell — Bacterial Spot | 17 | Tomato — Healthy |

### Crop Recommendation — Kaggle Dataset

| Property | Detail |
|---|---|
| Source | `atharvaingle/crop-recommendation-dataset` (Kaggle) |
| Local Path | `data/crop_recommendation/Crop_recommendation.csv` |
| Shape | 2,200 rows × 8 columns |
| Crop Classes | 22 |
| Features | N, P, K, temperature, humidity, ph, rainfall |

### Weather Data Source

- **Open-Meteo API** — free, no API key required
- Endpoint: `https://api.open-meteo.com/v1/forecast`
- Variables: `precipitation_sum`, `temperature_2m_max`, `temperature_2m_min`

---

## 4. Model Metrics

### Disease Detection (MobileNetV2)

On the current PlantVillage development test split (2,950 images):

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.63%** |
| **Macro F1-Score** | **97.19%** |
| **Test Loss** | 0.0707 |

#### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Apple — Apple Scab | 0.99 | 0.97 | 0.98 | 95 |
| Apple — Black Rot | 0.99 | 1.00 | 0.99 | 94 |
| Apple — Healthy | 0.99 | 0.99 | 0.99 | 248 |
| Corn — Gray Leaf Spot | 1.00 | 0.97 | 0.99 | 78 |
| Corn — Common Rust | 0.99 | 1.00 | 0.99 | 180 |
| Corn — Healthy | 1.00 | 1.00 | 1.00 | 175 |
| Grape — Black Rot | 1.00 | 1.00 | 1.00 | 177 |
| Grape — Healthy | 1.00 | 0.94 | 0.97 | 64 |
| Pepper Bell — Bacterial Spot | 0.97 | 0.99 | 0.98 | 151 |
| Pepper Bell — Healthy | 0.98 | 1.00 | 0.99 | 223 |
| Potato — Early Blight | 0.98 | 1.00 | 0.99 | 150 |
| Potato — Late Blight | 0.99 | 0.93 | 0.96 | 150 |
| Potato — Healthy | 0.85 | 0.96 | 0.90 | 24 |
| Tomato — Bacterial Spot | 0.99 | 0.98 | 0.99 | 320 |
| Tomato — Early Blight | 0.94 | 0.87 | 0.90 | 150 |
| Tomato — Late Blight | 0.96 | 0.94 | 0.95 | 287 |
| Tomato — Leaf Mold | 0.89 | 0.99 | 0.94 | 144 |
| Tomato — Healthy | 0.98 | 1.00 | 0.99 | 240 |
| **Macro Average** | **0.97** | **0.97** | **0.97** | **2950** |

These metrics do not represent performance on the separate field-condition dataset. That dataset must be evaluated independently.

### Crop Recommendation (Random Forest)

| Metric | Value |
|---|---|
| **Test Accuracy** | **99.55%** |
| **Macro F1-Score** | **0.9955** |
| Crop Classes | 22 |

---

## 5. Architecture Overview & Limitations

### Disease Detection — Model Architecture

```
Input Image (224 × 224 × 3)
        │
        ▼
┌──────────────────────────┐
│  MobileNetV2 Backbone    │
│  (ImageNet pre-trained)  │
└──────────────────────────┘
        │
        ▼
  GlobalAveragePooling2D
        │
        ▼
  Dense(18, softmax)
        │
        ▼
  Predicted Class + Confidence
```

### Training Strategy (Two-Phase)

| Phase | Layers Trained | Optimizer | Learning Rate | Max Epochs |
|---|---|---|---|---|
| Phase 1 — Head Training | Dense head only (base frozen) | Adam | Default | 3 (early stop) |
| Phase 2 — Fine-Tuning | Final 20 MobileNetV2 layers + head | Adam | 1e-5 | 10 (early stop) |

- **Checkpointing:** Best `val_accuracy` saved as `model/best_model.keras`
- **Early Stopping:** Patience = 2 on `val_accuracy`, restores best weights
- **Preprocessing:** `mobilenet_v2.preprocess_input` (scales to [-1, 1])
- **Augmentation (train only):** horizontal flip, rotation ±20°, zoom 20%, brightness [0.7–1.3], shift 10%

### Sustainability Score Formula

```
score = (water_score × 0.40) + (health_score × 0.40) + (resource_score × 0.20)

water_score    = max(0, 1 − water_used / max_water) × 100
health_score   = 0 if disease detected, 100 otherwise
resource_score = 100 − fertilizer_level

Badges:  >85 Platinum  |  >70 Green Guardian  |  <50 At Risk  |  else Needs Improvement
```

### Irrigation Decision Rules

| Priority | Condition | Recommendation |
|---|---|---|
| 1 | rain > 10 mm | Delay irrigation — heavy rain expected |
| 2 | rain > 4 mm + sandy soil | Light irrigation — sandy soil drains fast |
| 3 | rain > 4 mm + other soil | Skip irrigation — moderate rain sufficient |
| 4 | rain ≤ 4 mm + seedling | Irrigate gently — seedlings need moisture |
| 5 | rain ≤ 4 mm + other stage | Irrigate today — crop needs water |

### Known Limitations

| Limitation | Detail |
|---|---|
| Lab vs. Field gap | Trained on PlantVillage (studio images); field performance may drop |
| 18-class scope | Only Apple, Corn, Grape, Pepper, Potato, Tomato — others get generic fallback |
| Weather dependency | `src/weather.py` requires internet; offline fails with `ConnectionError` |
| Potato Healthy (F1 = 0.90) | Small support (24 test images) lowers this class's score |
| No IoT / GenAI | Bonus modules F (IoT), E (GenAI), G (Agentic Advisor) are out of scope |

---

## 6. Demo Video

> 🎬 **Demo video link:** *(To be added after recording)*

### Demo Script (~3–4 min)

| Timestamp | What to Show |
|---|---|
| 0:00 – 0:30 | App homepage — hero banner, module overview |
| 0:30 – 1:15 | Upload **diseased** leaf → disease label, confidence %, precaution steps |
| 1:15 – 1:45 | Upload **healthy** leaf → "Healthy Crop" status |
| 1:45 – 2:15 | Enter coordinates → Weather forecast + Irrigation advice |
| 2:15 – 2:50 | Enter farm inputs → Sustainability score + eco-badge |
| 2:50 – 3:20 | Enter soil/climate → Crop Recommendation output |
| 3:20 – 3:50 | CLI evaluation — confusion matrix + Macro-F1 |
| 3:50 – 4:00 | Close with team name & SIH 2026 branding |

---

## Project Structure

```text
AgriSmart-AI/
├── app.py                        # Streamlit unified web app
├── requirements.txt              # Python dependencies
├── check_data.py                 # Dataset integrity checker
│
├── model/
│   ├── train.py                  # MobileNetV2 transfer learning pipeline
│   ├── predict.py                # Single-image inference
│   ├── evaluate.py               # External dataset evaluator
│   ├── train_crop.py             # Crop recommendation RF training
│   ├── crop_predict.py           # Crop recommendation CLI prediction
│   ├── class_indices.json        # 18-class label mapping
│   ├── crop_rec_model.pkl        # Trained Random Forest model
│   ├── best_model.keras          # Best validation checkpoint
│   └── agrismart_model.keras     # Final saved disease model
│
├── src/
│   ├── precautions.py            # Disease precaution text (18 classes)
│   ├── weather.py                # Open-Meteo API weather forecast
│   ├── irrigation.py             # Rule-based irrigation advisor
│   ├── sustainability.py         # Formula-based sustainability scorer
│   └── crop_recommendation.py    # Crop recommender (soil + climate)
│
├── data/
│   ├── split/                    # train / val / test (PlantVillage)
│   └── crop_recommendation/      # Kaggle crop recommendation CSV
│
├── report/
│   ├── classification_report.txt # Per-class precision / recall / F1
│   ├── evaluation_metrics.json   # Accuracy + Macro-F1 + loss
│   └── confusion_matrix.png      # 18×18 confusion matrix heatmap
│
└── docs/
    ├── module_flowcharts.md      # Module interaction diagrams
    └── module_notes.txt          # Plain-English module explanations
```