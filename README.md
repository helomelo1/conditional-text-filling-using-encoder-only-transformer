# Text Infilling with Transformer Models

A simple text infilling system that learns to fill in masked tokens in sentences. Built with PyTorch and trained on IMDB movie reviews.

## What it does

The model takes sentences with masked tokens and learns to predict the original words:
```
Input:  "This [MASK] was [MASK]"
Output: "This movie was great"
```

## Project Structure
```
.
├── data.py          # Dataset loading and preprocessing
├── model.py         # Transformer-based infilling model
├── train.py         # Training script
├── requirements.txt # Dependencies
└── aclImdb/        # IMDB dataset (download separately)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download IMDB dataset:
```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

Your folder structure should look like:
```
project/
├── data.py
├── model.py
├── train.py
└── aclImdb/
    ├── train/
    │   ├── pos/
    │   └── neg/
    └── test/
        ├── pos/
        └── neg/
```

## Usage

### Test data loading
```bash
python data.py
```

### Test model
```bash
python model.py
```

### Train
```bash
python train.py
```

Training will:
- Run for 10 epochs
- Save checkpoints to `checkpoints/`
- Use MPS (Apple Silicon) or CPU
- Take ~2-3 hours on M3 MacBook

## Model Architecture

- Transformer encoder (BERT-style)
- 512 hidden dimensions
- 6 layers
- 8 attention heads
- ~42M parameters

## Results

Expected performance after training:
- Training accuracy: ~50-60%
- Validation accuracy: ~45-55%
- Loss decreases from ~10 to ~2-3

## Hyperparameters

Key settings in `train.py`:
- Batch size: 32
- Learning rate: 3e-4
- Mask ratio: 15%
- Max sequence length: 128
- Optimizer: AdamW with cosine decay

## Notes

- Uses IMDB reviews (25k train, 25k test)
- Removes HTML tags (`<br />`) during preprocessing
- Only computes loss on masked positions
- Gradient clipping at 1.0 for stability