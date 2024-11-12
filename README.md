# Image Retrieval System

This project implements various image retrieval methods including Color Histograms, PCA, BoVW, and CNN Classification.
Note: some of the functions are not implemented yet or may throw errors.

## Setup and Installation

### 1. Environment Setup

First, create and activate the virtual environment:

```bash
# Create virtual environment
python -m venv cvpr

# Activate virtual environment
# On Linux/Mac:
source cvpr/bin/activate
# On Windows:
.\cvpr\Scripts\activate
```

### 2. Install Dependencies

With the virtual environment activated, install required packages:

```bash
pip install -r requirements.txt
```

### 3. Running Individual Components

#### Color Histogram Retrieval
```bash
python src/color_histogram.py --query_path path/to/query/image
```

#### PCA-based Retrieval
```bash
python src/pca_retrieval.py --query_path path/to/query/image
```

#### BoVW Retrieval
```bash
python src/bovw_retrieval.py --query_path path/to/query/image
```

#### CNN Classification
```bash
python src/cnn_classifier.py --model resnet18 --batch_size 32
```

### 4. Running the Streamlit Application

To launch the interactive web interface:

```bash
# Activate environment if not already activated
source cvpr/bin/activate  # or .\cvpr\Scripts\activate on Windows

# Run Streamlit app
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501` by default.

### 5. Viewing Results

Results are saved in the `results/` directory with the following structure:
```
results/
├── color_histogram/
├── pca/
├── bovw/
└── cnn/
    ├── confusion_matrix.png
    ├── training_curves.png
    └── test_queries.png
```

### 6. Troubleshooting

If you encounter any issues:

1. Verify environment activation:
```bash
# Should show python from cvpr environment
which python  # on Linux/Mac
where python  # on Windows
```

2. Check CUDA availability for CNN:
```python
import torch
print(torch.cuda.is_available())
```

3. Verify package installation:
```bash
pip list
```
