# Transfer Learning for Cross-Disease Chest X-Ray Analysis and Classification 🫁

A comprehensive deep learning project utilizing transfer learning and ensemble methods to classify chest X-ray images across multiple respiratory diseases including COVID-19, bacterial pneumonia, viral pneumonia, and normal cases.

## 📋 Project Overview

This project implements and compares multiple state-of-the-art deep learning models for automated chest X-ray classification. By leveraging pre-trained ImageNet weights and ensemble techniques, the system achieves high accuracy in distinguishing between four distinct categories of chest conditions.

## 🏗️ Architecture

The project employs transfer learning with multiple convolutional neural network architectures, combining individual models into powerful ensemble classifiers for improved prediction accuracy and robustness.

**Model Pipeline:**
- Pre-trained models on ImageNet
- Fine-tuning on chest X-ray dataset
- Ensemble fusion techniques
- Softmax probability analysis
- Performance evaluation and visualization

## 🤖 Models Implemented

### Individual Models
1. **VGG16** - 16-layer deep convolutional network
2. **Xception** - Depthwise separable convolutions architecture
3. **DenseNet121** - Densely connected convolutional network

### Ensemble Models
1. **Ensemble: Xception + VGG16**
   - Combined predictions from Xception and VGG16
   - Weighted averaging of probability outputs

2. **Ensemble: Xception + DenseNet121**
   - Fusion of Xception and DenseNet121 predictions
   - Max softmax probability analysis
   - Confusion matrix visualization

## 📊 Classification Categories

The models classify chest X-rays into four distinct categories:

1. **COVID-19** - Coronavirus disease manifestations
2. **Normal** - Healthy chest X-rays
3. **Pneumonia (Bacterial)** - Bacterial infection cases
4. **Pneumonia (Viral)** - Viral infection cases

## 📁 Project Structure

```
Transfer-Learning-Chest-XRay/
├── notebooks/
│   ├── VGG16_Model.ipynb
│   ├── Xception_Model.ipynb
│   ├── DenseNet121_Model.ipynb
│   ├── Ensemble_Xception_VGG16.ipynb
│   └── Ensemble_Xception_DenseNet121.ipynb
├── data/
│   ├── COVID-19/
│   ├── NORMAL/
│   ├── PNEUMONIA_bacterial/
│   └── PNEUMONIA_viral/
├── models/                  # Saved model weights
├── results/
│   ├── confusion_matrices/
│   ├── performance_plots/
│   └── metrics/
├── requirements.txt
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- Google Colab account (for cloud execution)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Healer3504/Transfer-Learning-for-Cross-Disease-Chest-X-RAY-Analysis-and-Claasification.git
   cd Transfer-Learning-Chest-XRay
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries

```txt
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
```

## 📥 Dataset

**Source**: Kaggle Open Source Chest X-Ray Dataset

**Dataset Structure**:
- Total images: ~4000+ X-ray images
- Training set: 80%
- Validation set: 10%
- Test set: 10%

**Download Instructions**:
1. Visit [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets)
2. Download and extract to `data/` directory
3. Organize images into respective category folders

## 🖥️ Running the Models

### Google Colab (Recommended)

1. Upload notebooks to Google Colab
2. Mount Google Drive (if dataset stored there):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Run cells sequentially
4. Models will train and generate results automatically

### Local Execution

```bash
# Run individual model notebooks
jupyter notebook notebooks/VGG16_Model.ipynb

# Or use Python script
python train_model.py --model vgg16 --epochs 50 --batch_size 32
```

## 📈 Performance Metrics

The project evaluates models using comprehensive metrics:

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision scores
- **Recall**: Sensitivity for each category
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification matrix
- **ROC-AUC Curves**: Performance across thresholds
- **Max Softmax Probability**: Confidence analysis

### Visualization Components
- Training/Validation accuracy curves
- Loss curves over epochs
- Confusion matrices with heatmaps
- Class-wise performance bar charts
- Ensemble model comparison graphs
- Softmax probability distributions

## 🎯 Key Features

- **Transfer Learning**: Leverages pre-trained ImageNet weights
- **Multiple Architectures**: Comparison across VGG16, Xception, and DenseNet121
- **Ensemble Methods**: Combined model predictions for improved accuracy
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Confusion Matrix Analysis**: Visual representation of classification performance
- **Probability Analysis**: Max softmax probability for confidence assessment
- **Data Augmentation**: Enhanced training with image transformations
- **Class Balancing**: Handling imbalanced dataset scenarios

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| VGG16 | 81.47% | 80.25% | 80.41% | 79.90 |
| Xception | 82.59% | 80.67% | 81.39% | 80.26 |
| DenseNet121 | 82.84% | 83.12% | 80.92% | 80.92 |
| Ensemble (Xception + VGG16) | 83.82% | 83.35% | 83.82% | 83.19 |
| Ensemble (Xception + DenseNet121) | 80.25% | 80.17% | 80.25% | 78.41 |

*Note: Update values after training completion*

## 🛠️ Technology Stack

- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **Scikit-learn** - Metrics and preprocessing
- **OpenCV** - Image processing
- **Google Colab** - Cloud-based execution environment

## 💻 System Requirements

**For Local Execution**:
- RAM: 16GB minimum (32GB recommended)
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: 10GB+ free space
- OS: Windows/Linux/macOS

**For Google Colab**:
- Google account
- Stable internet connection
- GPU runtime enabled (Runtime → Change runtime type → GPU)

## 🔧 Troubleshooting

### Common Issues

**Issue**: Out of memory errors
```python
# Solution: Reduce batch size
batch_size = 16  # Instead of 32
```

**Issue**: Dataset not found
```bash
# Verify dataset path
import os
print(os.listdir('data/'))
```

**Issue**: Model training too slow
- Enable GPU in Google Colab
- Reduce image resolution
- Use smaller batch sizes

**Issue**: Poor model performance
- Increase training epochs
- Apply data augmentation
- Adjust learning rate
- Check class imbalance

## 📚 Model Training Tips

1. **Data Preprocessing**:
   - Normalize pixel values (0-1)
   - Resize images to model input size
   - Apply data augmentation

2. **Training Strategy**:
   - Use early stopping
   - Implement learning rate scheduling
   - Save best model checkpoints

3. **Ensemble Techniques**:
   - Average probability predictions
   - Weighted voting based on validation accuracy
   - Stack predictions as features

## 🔬 Future Enhancements

- [ ] Add more ensemble combinations
- [ ] Implement attention mechanisms
- [ ] Deploy as web application
- [ ] Add explainability (Grad-CAM)
- [ ] Expand to more disease categories
- [ ] Real-time prediction interface

## 📖 References

- VGG16: Simonyan & Zisserman (2014)
- Xception: Chollet (2017)
- DenseNet: Huang et al. (2017)
- Transfer Learning: Pan & Yang (2010)

## 👤 Author

**Maintained by**: [Healer3504](https://github.com/Healer3504)

## 📄 License

MIT License - Feel free to use this project for educational and research purposes.

## 🙏 Acknowledgments

- Kaggle for providing the open-source chest X-ray dataset
- TensorFlow and Keras teams for excellent deep learning frameworks
- Medical imaging research community for domain insights

---

**Note**: This project is for educational and research purposes. Not intended for clinical diagnosis without proper validation.
