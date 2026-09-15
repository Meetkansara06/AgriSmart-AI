# AgriSmart AI 🌱

AI-powered crop disease detection and smart farming advisory system.

**Team:** Digital Dynamos  
**Hackathon:** Smart India Hackathon (SIH) 2026 — Internal Round

Farmers need accessible crop disease identification and actionable guidance. AgriSmart AI combines MobileNetV2-based computer vision with weather intelligence, smart irrigation, sustainability scoring, crop recommendation, and a grounded multilingual Farmer Assistant — all in a single Streamlit application.

---

## Quick Start — Reproduce in Under 10 Minutes

> ⚠️ **Requires Python 3.10–3.13.** Python 3.14+ will fail (TensorFlow 2.21.0 incompatibility).

```powershell
git clone https://github.com/Meetkansara06/AgriSmart-AI.git
cd AgriSmart-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```powershell
python model\predict.py --image "path\to\leaf.jpg"       # Disease prediction
python -m streamlit run app.py                            # Web app → http://localhost:8501
python -m pytest tests/ -q                                # 27 tests passed
```

---

## SIH Problem Statement Alignment

| Capability | Technology | Status |
|---|---|---|
| Core — Disease Detection | MobileNetV2 / TensorFlow 2.21 | ✅ Implemented |
| Bonus A — Crop Recommendation | Random Forest / scikit-learn | ✅ Implemented |
| Bonus B — Smart Irrigation | Rule-based (rainfall + soil + stage) | ✅ Implemented |
| Bonus C — Weather Intelligence | Open-Meteo API (free, no key) | ✅ Implemented |
| Bonus D — Sustainability Score | Weighted formula (0–100) | ✅ Implemented |
| Bonus E — Farmer Assistant | Google Gemini (grounded GenAI) | ✅ Implemented |
| Bonus F — IoT Integration | — | ❌ Not implemented |
| Bonus G — Agentic Advisor | — | ❌ Not implemented |

---

## Core AI/ML — Disease Detection

**TensorFlow 2.21.0 · MobileNetV2 · ImageNet pretrained · 224×224 input · 18 classes (closed-set)**

```
Input (224×224×3) → MobileNetV2 (ImageNet) → GlobalAveragePooling2D → Dense(18, softmax) → Disease + Confidence
```

### Training Strategy

| Phase | What | Optimizer | LR | Epochs |
|---|---|---|---|---|
| 1 — Head | Dense head only (base frozen) | Adam | Default | 3 (early stop) |
| 2 — Fine-tune | Last 20 MobileNetV2 layers + head | Adam | 1e-5 | 10 (early stop) |

- **Checkpointing:** Best `val_accuracy` → `model/best_model.keras`
- **Early stopping:** Patience = 2, restores best weights
- **Augmentation (train only):** Flip, rotation ±20°, zoom 20%, brightness [0.7–1.3], shift 10%
- **Preprocessing:** `mobilenet_v2.preprocess_input` (pixels → [-1, 1])

### 18 Supported Classes

Apple (Scab, Black Rot, Healthy) · Corn (Gray Leaf Spot, Common Rust, Healthy) · Grape (Black Rot, Healthy) · Pepper Bell (Bacterial Spot, Healthy) · Potato (Early Blight, Late Blight, Healthy) · Tomato (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Healthy)

> This is a **closed-set classifier**. Images outside these 18 classes receive an incorrect existing-class label — the model has no "unknown" output.

---

## Dataset & Evaluation Distinction

### Disease — PlantVillage

~54,000 images filtered to 18 classes · **70/15/15** train/val/test split · seed=42 · `split-folders` v0.6.1  
Paths: `data/split/train/`, `data/split/val/`, `data/split/test/`

### Crop Recommendation — Kaggle

`atharvaingle/crop-recommendation-dataset` · 2,200 rows · 22 crop classes · 7 features (N, P, K, temp, humidity, pH, rainfall)  
Path: `data/crop_recommendation/Crop_recommendation.csv`

> ⚠️ **Field-Condition Evaluation Distinction**  
> The official SIH PS uses a **separate organizer-held-out field-condition test set** that is **not in this repository**. All metrics below are from the **internal PlantVillage development test split only**. Field-condition performance should only be reported after evaluation on that separate dataset.

---

## Results

### Disease Detection — PlantVillage Dev Test (2,950 images)

> **Internal development results only — not official field-condition performance.**

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.63%** |
| **Macro F1-Score** | **97.19%** |
| Test Loss | 0.0707 |

Detailed per-class report → `report/classification_report.txt`  
Confusion matrix heatmap → `report/confusion_matrix.png`  
Raw metrics → `report/evaluation_metrics.json`

### Crop Recommendation — Random Forest

| Metric | Value |
|---|---|
| **Accuracy** | **99.55%** |
| **Macro F1** | **0.9955** |

2,200 rows · 22 crop classes · 80/20 stratified split  
Report → `report/crop_recommendation_classification_report.txt`

---

## Bonus Modules

**A — Crop Recommendation:** Random Forest (n_estimators=100) using N, P, K, temperature, humidity, pH, rainfall → best crop. Artifact: `model/crop_rec_model.pkl`

**B — Smart Irrigation:** Rule-based advisory combining tomorrow's rainfall (from weather module), soil type (sandy/loamy/clay), and crop stage (seedling/growing/mature). Not an ML model.

**C — Weather Intelligence:** Open-Meteo 3-day forecast → tomorrow's precipitation, max/min temperature. Free, no API key, internet required.

**D — Sustainability Score:**
```
score = (water_score × 0.40) + (health_score × 0.40) + (resource_score × 0.20)
Badges:  >85 Platinum | >70 Green Guardian | <50 At Risk | else Needs Improvement
```

**E — Farmer Assistant (GenAI):** Gemini-powered conversational layer grounded on actual session results (disease, weather, irrigation, sustainability, crop). English + Gujarati. Does **not** diagnose, override modules, or invent data. Session-only history. Text-only (no voice).

---

## Farmer Assistant — Gemini Setup

All modules except the Farmer Assistant work **without an API key**. If no key is configured, the app runs normally — only the Assistant tab shows a setup message.

### Get a Gemini API Key

1. Open **https://aistudio.google.com/**
2. Sign in with a Google account
3. Create/select a project → create API key → copy it

### Configure

Create `.streamlit/secrets.toml` in the project root:

```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
```

Or set an environment variable: `$env:GEMINI_API_KEY = "YOUR_KEY_HERE"`

**Technical:** SDK `google-genai` · Default model `gemini-3.5-flash-lite` · Override via `GEMINI_MODEL` env var

> ⚠️ **Never** commit API keys to Git. `secrets.toml` is already in `.gitignore`. If exposed, revoke immediately in Google AI Studio.

---

## Project Structure

```
AgriSmart-AI/
├── app.py                         # Streamlit web app (5 tabs)
├── requirements.txt
├── model/
│   ├── train.py                   # MobileNetV2 training
│   ├── predict.py                 # Single-image inference
│   ├── evaluate.py                # External dataset evaluator
│   ├── train_crop.py              # Crop recommendation training
│   ├── crop_predict.py            # Crop recommendation CLI
│   ├── agrismart_model.keras      # Trained disease model
│   ├── crop_rec_model.pkl         # Trained Random Forest
│   └── class_indices.json         # 18-class label mapping
├── src/
│   ├── precautions.py             # Disease precautions
│   ├── weather.py                 # Open-Meteo API
│   ├── irrigation.py              # Rule-based irrigation
│   ├── sustainability.py          # Sustainability scorer
│   └── farmer_assistant.py        # Gemini assistant
├── data/
|   ├── split/                     # train / val / test
|   ├── crop_recommendation        # Crop_recommendation.csv                     
├── report/                        # Metrics, reports, confusion matrix
├── tests/                         # 27 automated tests
└── .streamlit/secrets.toml        # API keys (git-ignored)
```

---

## Known Limitations

**1. Lab vs. Real Field Gap** — Trained on PlantVillage (studio images). Real field photos with soil, glare, and clutter may yield lower accuracy.

**2. Closed-Set 18 Classes** — Only 6 crop types supported. Major Indian staples (Rice, Wheat, Cotton, Sugarcane) are unsupported. Unknown images receive an incorrect existing-class label.

**3. Softmax Overconfidence** — Out-of-scope images still get high-confidence predictions (e.g., 92%+) instead of "unknown". The `softmax` layer forces probabilities to sum to 100%.

**4. Internet Required** — Weather and Farmer Assistant need connectivity. Disease detection and crop recommendation work offline.

**5. Scope** — Crop recommendation uses 7 inputs only. Irrigation is rule-based advisory. Sustainability is formula-based. IoT and Agentic Advisor are not implemented. Voice is not implemented.

---

## Testing

```powershell
python -m pytest tests/ -v          # 27 tests, all mocked, no API key needed
```

---

## Demo Video

> 🎬 **Demo link:** *[ADD FINAL DEMO LINK HERE]*

**Flow:** App overview → diseased leaf → healthy leaf → weather + irrigation → sustainability → crop recommendation → Farmer Assistant (EN + GU) → closing

---

## Disclaimer

AgriSmart AI is **decision support**. It does not replace professional agricultural advice. Field accuracy may differ from reported PlantVillage metrics. Crop recommendations and irrigation advice are statistical/rule-based. The Farmer Assistant explains results but does not independently diagnose. **Never commit API keys to version control.**

---

*Made with 💚 for sustainable agriculture — SIH 2026 - Internal Round*