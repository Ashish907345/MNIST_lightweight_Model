# MNIST Digit Classification using PyTorch

This project implements a Convolutional Neural Network (CNN) to classify MNIST handwritten digits with 99% accuracy using PyTorch.

## 🏗️ Model Architecture
The model uses a modern CNN architecture with the following key features:
- Two convolutional blocks with batch normalization & DropOut
- Global Average Pooling (GAP)
- Dropout for regularization
- ~16K trainable parameters

## 📊 Training Results

| Epoch | Test Loss | Accuracy | Correct/Total |
|-------|-----------|----------|---------------|
| 1 | 0.1542 | 96.15% | 9615/10000 |
| 2 | 0.0842 | 97.67% | 9767/10000 |
| 3 | 0.0585 | 98.40% | 9840/10000 |
| 4 | 0.0583 | 98.19% | 9819/10000 |
| 5 | 0.0453 | 98.62% | 9862/10000 |
| 6 | 0.0362 | 98.85% | 9885/10000 |
| 7 | 0.0380 | 98.66% | 9866/10000 |
| 8 | 0.0287 | 99.09% | 9909/10000 |
| 9 | 0.0330 | 98.92% | 9892/10000 |
| 10 | 0.0289 | 99.00% | 9900/10000 |
| 11 | 0.0261 | 99.16% | 9916/10000 |
| 12 | 0.0290 | 99.04% | 9904/10000 |
| 13 | 0.0266 | 99.02% | 9902/10000 |
| 14 | 0.0245 | 99.20% | 9920/10000 |
| 15 | 0.0221 | 99.31% | 9931/10000 |
| 16 | 0.0245 | 99.25% | 9925/10000 |
| 17 | 0.0228 | 99.27% | 9927/10000 |
| 18 | 0.0229 | 99.35% | 9935/10000 |
| 19 | 0.0231 | 99.29% | 9929/10000 |

## 📈 Key Observations
- **Best Accuracy**: 99.35% (Epoch 18)
- **Final Accuracy**: 99.29% (Epoch 19)
- **Convergence**: Model achieves >99% accuracy by epoch 8
- **Loss Trend**: Steady decrease in test loss from 0.1542 to 0.0231

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- PyTorch 1.7+
- Other dependencies in requirements.txt

