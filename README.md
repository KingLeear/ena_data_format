# ENA End-to-End Pipeline (Streamlit)

This repository provides an end-to-end pipeline for transforming raw student text into ENA-ready binary matrices, supporting segmentation, concept prediction, and network analysis.

## Overview

The pipeline supports the following workflow:
	1.	Upload raw textual data (CSV)
	2.	Segment text into analysis units (sentences)
	3.	Train or load a multiclass concept classifier
	4.	Generate concept predictions
	5.	Binarize predictions for ENA
	6.	Export ENA-ready data for network analysis

The system is designed for educational discourse analysis, particularly for epistemic or conceptual coding.

⸻

### Features
	•	Concept-driven paradigm generation (via OpenAI)
	•	Multiclass text classification (HuggingFace Transformers)
	•	Automatic sentence segmentation (English / Chinese / auto)
	•	ENA-ready binary encoding output
	•	Streamlit UI with progress feedback and data previews
	•	Model download and reuse

⸻

### Requirements
	•	Python 3.11 or 3.12 recommended
	•	macOS / Linux / Windows (macOS tested)
	•	pip

⸻

### Installation

```
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```


⸻

### Run the App

```streamlit run app.py```

Open in browser: http://localhost:XXXX

⸻

### OpenAI API Key

Enter your API key in the sidebar UI, or set it as an environment variable:

export OPENAI_API_KEY="your-key-here"


⸻

1. Concept CSV Format

Upload a CSV containing concept definitions.

#### Recommended (3 columns)

| code | label | definition |
|------|-------|------------|
| C1 | claim | A claim is a contestable statement… |
| C2 | position | A position is a writer’s stance… |
Alternative (2 columns)

#### Alternative (2 columns)

| concept | definition |
|---------|------------|
| claim | A claim is… |
| position | A position is… |


⸻

2. Raw Data CSV Format

Must contain:
	•	A student identifier column (e.g. student_id)
	•	A text column (e.g. comment_body)

Example:

student_id	comment_body
S001	I first re-read the question…
S002	This reminded me of last class…


⸻

3. Pipeline Steps

Step 1 — Generate Paradigms and Train Model
	•	Select paradigm language (zh/en)
	•	Choose number of paradigms per concept (recommend ≥30)
	•	Choose backbone model (bert / roberta)
	•	Train model

Output:
	•	data/_tmp_paradigms.csv
	•	Trained model directory (e.g. model_out/)

⸻

Step 2 — Upload Raw Data
	•	Select text column
	•	Select ID columns
	•	Select segmentation language

⸻

Step 3 — Run Full Pipeline

Produces:

| File | Description |
|------|-------------|
| _tmp_raw.csv | Raw uploaded data |
| _tmp_units.csv | Segmented sentence units |
| _tmp_pred.csv | Predicted concept probabilities |
| _tmp_ena.csv | ENA-ready binary matrix |


⸻

Output

The final ENA CSV contains:
	•	student_id
	•	row / unit identifiers
	•	sentence text
	•	binary concept indicators (0/1)

Example:

| student_id | text | C1 | C2 |
|------------|------|----|----|
| S001 | I re-read the question | 1 | 0 |
| S002 | This reminded me of last class | 0 | 1 |


⸻

### Model Download / Upload

After training you can:
	•	Download the model as a ZIP archive
	•	Upload an existing trained model ZIP to reuse

⸻

### Training Metrics

After training, metrics are displayed as a table:
	•	train_loss
	•	eval_loss
	•	eval_accuracy
	•	eval_f1_macro
	•	runtime

⸻

### Troubleshooting

Error: The least populated classes in y have only 1 member

Cause: A concept has too few training examples to stratify train/validation split.

Fix:
	•	Increase “paradigms per concept” (≥30 recommended)
	•	Or disable stratified splitting when counts < 2 (already handled automatically)

⸻

Error: No segmented units

Check:
	•	Correct text column selected
	•	min_len too large
	•	segmentation language mismatch

⸻

Error: No predictions generated

Check:
	•	_tmp_units.csv exists and contains text
	•	Model directory exists and is valid

⸻

Folder Structure

ena-tool/
├── app.py
├── ena_tool.py
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── _tmp_raw.csv
│   ├── _tmp_short_raw.csv
│   ├── _tmp_units.csv
│   ├── _tmp_pred.csv
│   └── _tmp_ena.csv
│
├── models/
│   └── <model_dir>/
│       ├── label_map.json
│       └── (HF model files...)
│
├── r/
│   ├── run_ena.R
│   ├── ena_functions.R
│   └── launch_shiny_app.R
│
├── outputs/
│   ├── ena_output_latest.csv
│   ├── ena_config_latest.json
│   └── ena_set_latest.RData
│
└── data_structure.md


⸻

Notes
	•	For reliable training, ensure each class has at least 20–30 samples.
	•	Paradigm quality directly affects model performance.
	•	ENA binarization supports both top1 and threshold modes.

