# AgriSmart AI — Model Report

**Team:** C-433, L.J. Institute of Engineering & Technology · **Hackathon:** SIH 2026 (Internal) · **Date:** September 2026

## 1. Task

Classify crop leaf images into 18 disease/healthy classes across 6 crops (Apple, Corn, Grape, Pepper Bell, Potato, Tomato). The system accepts a single leaf photograph and returns a predicted class label with confidence score. This is a closed-set classifier — it has no "unknown" output class.

## 2. Dataset & Split

| Property | Value |
|---|---|
| Dataset | PlantVillage (~54,000 images, filtered to 18 classes) |
| Split | 70% train / 15% validation / 15% test |
| Seed | 42 (`split-folders` v0.6.1) |
| Image size | 224 × 224 × 3 |
| Dev test set | 2,950 images |

> ⚠️ The official SIH evaluation uses a **separate organizer-held-out field-condition test set** (PlantDoc-style real-world images). That dataset is **not** in this repository and was **not** used for training or model selection. All metrics in this report are from the **internal PlantVillage development test split only**.

## 3. Model / Approach

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (ImageNet pretrained) → GlobalAveragePooling2D → Dense(18, softmax) |
| Framework | TensorFlow 2.21.0 |
| Phase 1 | Train classification head only (backbone frozen), Adam, 3 epochs max |
| Phase 2 | Fine-tune last 20 MobileNetV2 layers + head, Adam, LR=1e-5, 10 epochs max |
| Augmentation | Train-only: flip, rotation ±20°, zoom 20%, brightness [0.7–1.3], shift 10% |
| Checkpoint | Best validation accuracy → `model/best_model.keras` |
| Early stopping | Patience = 2 on val accuracy, restores best weights |
| Preprocessing | `mobilenet_v2.preprocess_input` (pixels → [-1, 1]) |

## 4. Results — PlantVillage Development Test (2,950 images)

| Metric | Value |
|---|---|
| **Accuracy** | **97.63%** |
| **Macro F1-Score** | **97.19%** |
| Test Loss | 0.0707 |

> These are **development-set results**. They must not be interpreted as the official SIH field-condition score. The official score is determined by the organizer using the separate held-out field test set.

## 5. Per-Class Precision & Recall

| Class | P | R | Class | P | R |
|---|---|---|---|---|---|
| Apple — Scab | 0.99 | 0.97 | Pepper Bell — Healthy | 0.98 | 1.00 |
| Apple — Black Rot | 0.99 | 1.00 | Potato — Early Blight | 0.98 | 1.00 |
| Apple — Healthy | 0.99 | 0.99 | Potato — Late Blight | 0.99 | 0.93 |
| Corn — Gray Leaf Spot | 1.00 | 0.97 | Potato — Healthy | 0.85 | 0.96 |
| Corn — Common Rust | 0.99 | 1.00 | Tomato — Bacterial Spot | 0.99 | 0.98 |
| Corn — Healthy | 1.00 | 1.00 | Tomato — Early Blight | 0.94 | 0.87 |
| Grape — Black Rot | 1.00 | 1.00 | Tomato — Late Blight | 0.96 | 0.94 |
| Grape — Healthy | 1.00 | 0.94 | Tomato — Leaf Mold | 0.89 | 0.99 |
| Pepper Bell — Bact. Spot | 0.97 | 0.99 | Tomato — Healthy | 0.98 | 1.00 |

Full report: `report/classification_report.txt` · Confusion matrix: `report/confusion_matrix.png`

## 6. Baseline Comparison

No explicit baseline Macro-F1 was provided in the official SIH Problem Statement materials available in this repository. Our PlantVillage development-test Macro-F1 is **0.9719**. Official field-test comparison is pending because the organizer-held-out field-condition score is not yet available.

## 7. Limitations

- **Lab vs. field gap:** PlantVillage images are studio-quality; real field photos (soil, glare, shadows) may produce lower accuracy.
- **Closed-set 18 classes:** The model cannot reject unknown crops/diseases — out-of-scope images still receive a high-confidence existing-class prediction due to `softmax`.
- **Limited crop coverage:** Only Apple, Corn, Grape, Pepper Bell, Potato, and Tomato are supported. Major Indian staples (Rice, Wheat, Cotton) are not.
- **Development metrics only:** The reported 97.19% Macro-F1 is from the PlantVillage development split. The official SIH field-condition test set was not used for training, validation, or model selection, and its score is not available in this repository.