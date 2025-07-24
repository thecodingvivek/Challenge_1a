# PDF Structure Detection System (Connecting the dots...)

**Ultra-Enhanced Version** - Advanced PDF structure detection with machine learning and heuristic approaches.

**Current Performance**: 44.40% average ground truth accuracy (Training: 88.89%, Individual files: 60.0%, 58.0%, 93.3%, 10.7%, 0.0%)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python run_system.py --mode train
```

### 3. Process a PDF

```bash
python run_system.py --mode process --input path/to/document.pdf --output result.json
```

### 4. Evaluate Performance

```bash
python run_system.py --mode evaluate
```

## 📁 Project Structure

```
.
├── run_system.py                    # Main entry point
├── src/
│   ├── standalone_json_generator.py           # ⭐ MAIN: Unified ML + heuristic processor
│   ├── ultra_enhanced_feature_extractor.py   # ⭐ Advanced feature extraction (40+ features)
│   ├── config_manager.py                      # Configuration management
│   └── api_wrapper.py                         # API utilities
├── scripts/
│   └── training/
│       ├── train_enhanced_model.py             # Training pipeline
│       └── prepare_data.py                     # Data preparation
│   └── evaluation/
│       └── comprehensive_evaluation.py         # System evaluation
├── data/
│   ├── raw_pdfs/                               # Training and test PDFs
│   ├── ground_truth/                          # Manual annotations
│   └── processed/                             # Processed features
├── models/
│   └── production/
│       ├── ultra_accuracy_optimized_classifier.pkl    # Trained model
│       └── ultra_accuracy_optimized_classifier_metadata.json
└── final_accuracy_report.py                   # Performance analysis
```

## 🎯 System Features

### Advanced Feature Engineering (40+ Features)

- **Font Analysis**: Size ratios, emphasis scoring, style detection
- **Pattern Recognition**: Title patterns, heading patterns
- **Position Features**: Bounding box analysis, page positioning
- **Text Analysis**: Length ratios, word counts, capitalization
- **Contextual Features**: Surrounding element analysis

### Machine Learning Pipeline

- **7-Model Ensemble**: RandomForest, XGBoost, Neural Networks, Extra Trees, Gradient Boosting, LightGBM, SVM
- **Training Accuracy**: 88.89%
- **Ground Truth Accuracy**: 44.40% average
- **Advanced Text Similarity**: Multi-algorithm matching with SequenceMatcher, Jaccard similarity

### Hybrid Approach

- ML predictions with heuristic fallbacks
- Confidence scoring and validation
- Enhanced heading detection (7-20 headings per document vs previous 0)

## 📊 Performance Analysis

| Test File             | Accuracy | Title Match | Headings Found |
| --------------------- | -------- | ----------- | -------------- |
| STEMPathwaysFlyer.pdf | 60.0%    | Found       | 5/4            |
| E0CCG5S239.pdf        | 58.0%    | Perfect     | 1/0            |
| E0CCG5S312.pdf        | 93.3%    | Good        | 17/17          |
| E0H1CM114.pdf         | 10.7%    | Found       | 4/39           |
| TOPJUMP-PARTY.pdf     | 0.0%     | Found       | 3/1            |

**Average**: 44.40% (Target: 90%) | **Title Detection**: 5/5 files ✅

## 🔧 Advanced Usage

### Direct Processing

```python
from src.standalone_json_generator import UltraEnhancedJSONGenerator

generator = UltraEnhancedJSONGenerator("models/production/ultra_accuracy_optimized_classifier.pkl")
result = generator.process_pdf("document.pdf")
print(result)
```

### Custom Training

```python
from scripts.training.train_enhanced_model import train_enhanced_model

# Train with custom data
success = train_enhanced_model()
```

## 🎯 Optimization Roadmap (To 90% Target)

### Immediate Improvements

1. **Semantic Embeddings**: BERT-based text similarity (+10-15% accuracy)
2. **Document-Type Models**: Specialized models for forms/technical docs/invitations
3. **Enhanced Preprocessing**: Better text normalization and formatting handling
4. **Confidence Thresholds**: Fine-tuned ML vs heuristic decision boundaries
5. **Training Data Expansion**: More balanced samples across document types

### Advanced Improvements

1. **Multi-modal Features**: Image and layout analysis
2. **Contextual Embeddings**: Document-aware feature engineering
3. **Active Learning**: Iterative model improvement with feedback
4. **Ensemble Optimization**: Advanced voting and weighting strategies

## 🔍 Key Components

### Ultra-Enhanced Feature Extractor

- 40+ engineered features across 5 categories
- Font-based analysis with emphasis detection
- Pattern matching for titles and headings
- Statistical normalization and data type handling

### Ultra-Enhanced JSON Generator

- Unified ML + heuristic processing
- Multi-strategy title and heading detection
- Confidence scoring with graceful fallbacks
- Enhanced structured output generation

### Training Pipeline

- Advanced text similarity with multiple algorithms
- Comprehensive accuracy calculation with precision/recall
- Detailed ground truth comparison and analysis
- Performance tracking and optimization guidance

## 📋 Requirements

- Python 3.8+
- PyMuPDF (fitz)
- scikit-learn
- xgboost
- lightgbm
- pandas
- numpy

## 📈 Recent Improvements

**Version History**:

- v1.0: Basic system (0% ground truth accuracy)
- v2.0: Enhanced features and ML (35% accuracy)
- v3.0: Ultra-enhanced system (44.40% accuracy, 100% title detection)

**Key Achievements**:

- ✅ Advanced text similarity implementation
- ✅ 40+ feature engineering with font analysis
- ✅ 7-model ensemble training (88.89% training accuracy)
- ✅ Comprehensive ground truth evaluation system
- ✅ Hybrid ML+heuristic approach with confidence scoring
- ✅ Detailed performance reporting and optimization pathway

## 🤝 Contributing

The system has a clear pathway to 90% accuracy through semantic embeddings and document-specific optimization. Key areas for contribution:

1. Semantic similarity implementation
2. Document type classification
3. Enhanced preprocessing pipelines
4. Training data augmentation

## 📄 License

See LICENSE file for details.
