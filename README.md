# IntelliStruct PDF Parser - Adobe Hackathon 2025 (Connecting the dots...)

**Challenge**: Adobe India Hackathon - Challenge 1A: Understand Your Document  
**Team**: `dot`  
**Repo**: `https://github.com/thecodingvivek/dot.git`

-----

## 📖 Project Overview

**IntelliStruct PDF Parser** is an advanced solution for the "Connecting the Dots" challenge. It transforms standard PDFs into structured, machine-readable outlines by intelligently identifying the document's title and hierarchical headings (H1, H2, H3).

Our system leverages a powerful **hybrid engine**, combining a sophisticated Machine Learning ensemble with a robust heuristic-based analyzer. This dual approach ensures both high accuracy on complex documents and blazing-fast performance, all while operating completely offline within a lightweight Docker container.

### ✨ Key Features

  * **Hybrid ML & Heuristic Engine**: Fuses a multi-model ML ensemble (XGBoost, RandomForest, etc.) with fine-tuned pattern recognition for superior accuracy.
  * **Advanced Feature Engineering**: Utilizes over 40 features, including font metrics, text patterns, positional data, and contextual cues from neighboring text blocks.
  * **Document-Aware Processing**: Intelligently detects the document type (e.g., `academic`, `business`, `form`) to apply specialized extraction rules.
  * **High Performance**: Processes a 50-page PDF in under 10 seconds, meeting strict performance constraints.
  * **Fully Dockerized & Offline**: Encapsulated in a `linux/amd64` Docker image with no external network dependencies, ensuring seamless evaluation.

-----

## 🛠️ Our Approach & Methodology

Our solution follows a multi-stage pipeline, designed for maximum accuracy and efficiency.

\<p align="center"\>
\<img src="architecture.png" alt="System Architecture Diagram" width="800"\>
\</p\>

1.  **Text Block Extraction (PyMuPDF)**: We start by extracting rich, detailed information from the PDF using `PyMuPDF (fitz)`. Instead of just text, we capture metadata for each text block, including font type, size, weight (bold), flags, and precise coordinates (bbox).

2.  **Advanced Feature Engineering**: This is the core of our system. For each text block, we compute a wide array of features to understand its role in the document:

      * **Font Features**: `avg_font_size`, `font_size_ratio` (relative to the document average), `has_bold`.
      * **Positional Features**: `y_position`, `is_top_quarter`, `relative_position`.
      * **Textual & Pattern Features**: `word_count`, `all_caps_ratio`, `starts_with_number`, and scores from custom `regex` patterns designed to find titles and headings.
      * **Contextual Features**: We analyze the relationship between a block and its neighbors (`prev_font_size`, `next_text_length`) to understand the document flow.

3.  **Hybrid Classification (ML + Heuristics)**:

      * **ML Ensemble**: We use an ensemble of powerful classifiers, including `XGBoost`, `RandomForest`, `LightGBM`, and `ExtraTreesClassifier`. This variety prevents overfitting and captures different types of patterns. The model is trained on a labeled dataset to classify each block as `Title`, `H1`, `H2`, `H3`, or `Paragraph`.
      * **Heuristic Engine**: A parallel engine uses document-aware rules and pattern matching. It serves as both a rapid baseline and a validation layer for the ML predictions.
      * **Intelligent Decision Making**: The system dynamically decides whether to trust the ML prediction or the heuristic rule for each block based on the ML model's confidence score and the block's characteristics. This hybrid approach is key to our high accuracy.

4.  **Hierarchical Outline Generation**: Once all blocks are classified, we perform a final post-processing pass to ensure structural integrity. This includes:

      * Selecting the most probable `Title`.
      * Ensuring logical heading order (e.g., an H2 follows an H1).
      * Removing duplicate or false-positive headings.
      * Formatting the final output into the required JSON structure.

-----

## 🔧 Models and Libraries

  * **Core Libraries**:
      * `PyMuPDF (fitz)`: For robust PDF parsing.
      * `scikit-learn`: For our ML model ensemble, feature scaling, and evaluation.
      * `xgboost`, `lightgbm`: For high-performance gradient boosting models.
      * `pandas`, `numpy`: For efficient data manipulation.
  * **ML Models**:
      * Our primary classifier is an **Ensemble Model** that includes:
          * `RandomForestClassifier`
          * `ExtraTreesClassifier`
          * `XGBClassifier`
          * `LGBMClassifier`
          * `GradientBoostingClassifier`
          * And others, whose predictions are weighted by their individual accuracy.

-----

## 📁 Project Structure

The repository is organized to separate data, source code, and outputs clearly.

```
dot/
├── data/                  # Labeled data for training and evaluation
│   ├── ground_truth/
│   │   ├── test/          # Ground truth JSONs for the test set
│   │   └── training/      # Ground truth JSONs for the training set
│   ├── raw_pdfs/
│   │   ├── test/          # PDFs for testing the model
│   │   └── training/      # PDFs for training the model
│   └── processed/         # Stores intermediate files like extracted features
├── input/                 # Directory for placing input PDFs for processing
├── models/                # Stores the trained ML model
│   └── production/
│       └── ultra_accuracy_optimized_classifier.pkl
├── output/                # Directory where output JSONs are saved
├── src/                   # Source code for the project
│   ├── feature_extractor.py  # Extracts features from PDF blocks
│   ├── json_generator.py     # Core hybrid classification logic
│   └── train_model.py        # Training and evaluation pipeline
├── .dockerignore          # Specifies files to exclude from the Docker build
├── .gitignore
├── Dockerfile             # Defines the container for the application
├── README.md              # You are here!
├── requirements.txt       # Lists all Python dependencies
└── run_system.py          # Main executable script for the system
```

-----

## 📊 Performance & Results

Our solution is optimized to adhere to all hackathon constraints.

  * **Execution Time**: **\~[XX] seconds** for a 50-page document (well within the $\\le10$ second limit).
  * **Model Size**: The serialized model is **\~[XX] MB** (well under the $\\le200$ MB limit).
  * **Runtime**: Runs entirely on **CPU** with no GPU dependencies and is compatible with `linux/amd64` architecture.
  * **Accuracy**: Our detailed evaluation script (`src/train_model.py`) measures accuracy using **Precision, Recall, and F1-Score** against the ground truth. Our current score on the provided test set is **[Your Accuracy Score]%**.

-----

## 🚀 How to Build and Run

### Prerequisites

  * [Docker](https://www.google.com/search?q=https://www.docker.com/get-started) must be installed and running.

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/thecodingvivek/dot.git
    cd dot
    ```

2.  **Prepare Input Files**

      * Create an `input` directory in the project root.
      * Place all your PDF files (`.pdf`) inside the `input` directory.

    <!-- end list -->

    ```bash
    mkdir input
    cp path/to/your/document.pdf input/
    ```

3.  **Build the Docker Image**

      * Run the following command from the project root. Replace `mysolutionname` with some name.

    <!-- end list -->

    ```bash
    docker build --platform linux/amd64 -t mysolutionname:latest .
    ```

4.  **Run the Solution**

      * The following command will process all PDFs from the `input` directory and save the corresponding `.json` files in an `output` directory.

    <!-- end list -->

    ```bash
    docker run --rm \
      -v $(pwd)/input:/app/input \
      -v $(pwd)/output:/app/output \
      --network none \
      mysolutionname:latest
    ```

      * The results will appear in the `output` folder in your project directory.

### Development Workflow: Training & Evaluation

Our project includes scripts for a full development cycle. Training is a necessary step to build the classification model from scratch using the provided labeled data.

  * **To run training:**
      * This command processes the labeled data, trains the ML ensemble, and saves the final model to the `models/production/` directory.
    ```bash
    # (Inside a running container or after installing dependencies locally)
    python3 run_system.py --mode train
    ```
  * **To run evaluation (Optional):**
      * This command evaluates the trained model against the ground truth test set and provides a detailed accuracy report.
    <!-- end list -->
    ```bash
    # (Inside a running container or after installing dependencies locally)
    python3 run_system.py --mode evaluate
    ```