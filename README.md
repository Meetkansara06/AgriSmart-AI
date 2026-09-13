# AgriSmart-AI
AI-powered crop disease detection and smart agriculture advisory system for sustainable farming.

 # AgriSmart AI
  ## 1. Modules Built
  ## 2. Setup & Run Instructions
  ## 3. Dataset Used
## AgriSmart AI

AI-powered crop disease detection and smart agriculture advisory system for sustainable farming.

## Current Status

The Core ML pipeline uses TensorFlow 2.21.0 and MobileNetV2 pretrained on ImageNet. It includes:

- Training-only image augmentation
- Official MobileNetV2 preprocessing
- Frozen-head training followed by fine-tuning of the final 20 base-model layers
- Validation-based checkpointing and early stopping
- Accuracy, Macro-F1, classification report, and confusion matrix evaluation
- A standalone image prediction interface

## Advisory Modules

The following modules are pending explicit functional requirements and are intentionally unchanged:

- `src/irrigation.py`
- `src/precautions.py`
- `src/sustainability.py`
- `src/weather.py`

No irrigation, treatment, pesticide, fertilizer, weather-advisory, or integration logic has been inferred or added.

## Setup

Create or activate the existing Windows virtual environment, then install dependencies:

```powershell
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

Run from the project root:

```powershell
\.venv\Scripts\python.exe model\train.py
```

The script reuses `data/split` when it already exists. It does not recreate or overwrite the existing split.

Training outputs are saved to `model/` and `report/`.

## Predict One Image

```powershell
\.venv\Scripts\python.exe model\predict.py --image "path\to\leaf.jpg"
```

The command prints the predicted disease class and confidence.

## Evaluate an Unseen Dataset

The external evaluator expects one subdirectory per class and never trains on the supplied data:

```powershell
\.venv\Scripts\python.exe model\evaluate.py --data-dir "path\to\external_dataset"
```

Reports are written to `report/external_evaluation/` by default. Use the organizer or field-condition dataset here only for final evaluation. Do not add it to the training split.

## Dataset

The current development dataset is PlantVillage, split into:

- `data/split/train`
- `data/split/val`
- `data/split/test`

The organizer or field-condition test dataset is reserved for measuring real-world generalization and is not used during training or model selection.

## Development Metrics

On the current PlantVillage development test split:

- Test accuracy: `97.63%`
- Macro-F1: `97.19%`

These metrics do not represent performance on the separate field-condition dataset. That dataset must be evaluated independently.

## Crop Recommendation

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

Train the module with:

```powershell
.\.venv\Scripts\python.exe model\train_crop.py --data "data\crop_recommendation\Crop_recommendation.csv"
```

Run a prediction with:

```powershell
.\.venv\Scripts\python.exe model\crop_predict.py --N 90 --P 42 --K 43 --temperature 25 --humidity 80 --ph 6.5 --rainfall 200
```

The Streamlit Crop Recommendation tab accepts the same seven inputs and calls
the saved model independently of the disease image workflow. The result is a
data-driven recommendation, not a guarantee of field performance; soil tests,
local agronomy, weather, and farmer expertise remain important.

## Project Structure

```text
AgriSmart-AI/
├── data/split/       # Development train, validation, and test data
├── model/            # Training, prediction, evaluation, and saved models
├── report/           # Evaluation reports and confusion matrices
├── src/              # Agriculture advisory modules
├── check_data.py
├── requirements.txt
└── README.md
```
  ## 4. Model Metrics
  ## 5. Architecture Overview & Limitations
  ## 6. Demo Video
  