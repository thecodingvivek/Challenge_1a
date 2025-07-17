# Adobe PDF Document Structure Classifier

This project intelligently detects document structure from PDF files using classical ML and NLP-based feature extraction. It classifies text blocks into structural labels like `Title`, `H1`, `H2`, `H3`, and `Paragraph`.

Built for the Adobe India Hackathon, the pipeline is designed to run fully offline and CPU-optimized using PyMuPDF and LightGBM — no deep learning or GPU dependencies.

---

## Features

- PDF parsing with PyMuPDF (`fitz`)
- Visual + spatial layout feature extraction
- NLP and linguistic signal-based feature engineering
- Classical ML model training (LightGBM)
- JSON output generation in Adobe's expected format
- Fully Docker-compatible

---

## 📂 Directory Structure

```

adobe-pdf-structure-ai/
├── samples/             # Sample PDFs and labeled JSONs
├── heading_classifier.pkl
├── feature_extractor.py     # Extracts layout + linguistic features
├── json_generator.py        # CLI entrypoint: training + prediction
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image config
└── README.md            # You're here.

````

---

## ⚙️ Installation

### 📦 Install with pip (for local testing)

```bash
pip install -r requirements.txt
````

Required packages:

* `pymupdf`
* `pandas`
* `scikit-learn`
* `lightgbm`
* `nltk`
* `joblib`

### 🐳 Build Docker image (optional)

```bash
docker build -t adobe-doc-structure .
```

---

## 🧪 How to Use

### 🔧 Training the Model

Train using `.pdf` and `.json` pairs stored in the same folder (`samples/`):

```bash
python src/json_generator.py --mode train --input samples --output samples --model models/heading_classifier.pkl
```

* Each `.pdf` should have a corresponding `.json` with the same name (e.g., `sample1.pdf` and `sample1.json`)
* Trained model will be saved to `models/heading_classifier.pkl`

---

### 🧠 Predicting Structure of New PDFs

```bash
python src/json_generator.py --mode predict --input samples --output samples --model models/heading_classifier.pkl
```

* Outputs will be saved as `.json` in the output directory (same structure as Adobe expects)

---

## 🗃️ JSON Output Format

```json
{
  "title": "Main Title of the Document",
  "outline": [
    { "level": "H1", "text": "Section Title", "page": 1 },
    { "level": "H2", "text": "Subsection", "page": 2 },
    ...
  ]
}
```

---

## 📈 Current Limitations

* The model is currently trained on limited examples (initial version)
* Low support for edge cases (e.g., overlapping headings, noisy formatting)
* Future updates may include:

  * Custom n-gram heuristics
  * Better handling of visual hierarchy
  * Additional post-processing on outputs