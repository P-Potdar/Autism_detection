Here's a more detailed and comprehensive `README.md` for your project. This version covers everything from setup to troubleshooting, giving users a clear understanding of the project's goals, structure, and instructions.

---

# Autism Detection Using CNN with DenseNet121

This project applies deep learning techniques to detect autism from images using a Convolutional Neural Network (CNN) based on DenseNet121. The goal is to classify images into two categories: autistic and non-autistic, leveraging transfer learning and enhanced data augmentation to achieve reliable performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Evaluation and Results](#evaluation-and-results)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Overview

This project leverages transfer learning with DenseNet121, a deep neural network architecture pretrained on ImageNet, to classify autism-related images. By freezing certain layers in DenseNet121, we capture essential image features while fine-tuning the model for autism classification. The model is trained on augmented data to improve generalization, and learning rate scheduling is applied for optimized performance.

### Objective
- To create a deep learning model capable of detecting autism from images by identifying significant patterns in visual data.

## Features

- **Transfer Learning** with DenseNet121.
- **Enhanced Data Augmentation** for improved model generalization.
- **Fine-tuning Layers** in DenseNet121 for optimized results.
- **Dynamic Learning Rate Scheduling** to handle plateaus in model training.
- **Automated Classification Report Generation** post-model training.

## Project Structure

```
├── consolidated                   # Dataset directory containing images
│   ├── autism                     # Folder with images labeled as autistic
│   ├── non_autism                 # Folder with images labeled as non-autistic
├── cnn.py                         # Main script to preprocess data, build and train model
├── requirements.txt               # Project dependencies
├── README.md                      # Project README
├── models                         # Folder to save trained model checkpoints
└── plots                          # Directory for saving accuracy and loss plots
```

## Requirements

To replicate this project, you’ll need the following libraries:

- Python 3.8+
- TensorFlow
- NumPy
- scikit-learn
- Matplotlib

The dependencies are listed in `requirements.txt`. 

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/autism-detection-cnn.git
   cd autism-detection-cnn
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Place your dataset in the `consolidated/` directory with two subdirectories:

- `autism/` – contains images classified as autistic
- `non_autism/` – contains images classified as non-autistic

Ensure that each image category has enough samples for model training and validation.

## Training the Model

To train the model, run the following command:

```bash
python cnn.py
```

This script:

1. Sets up the dataset for training and validation using `ImageDataGenerator` with data augmentation.
2. Defines the CNN model architecture using DenseNet121 as the base.
3. Compiles and trains the model, using callbacks for learning rate adjustment.
4. Saves model checkpoints in the `models` directory.

### Important Parameters in `cnn.py`
- **Image Dimensions**: 128x128 pixels.
- **Batch Size**: 32 images per batch.
- **Epochs**: Configurable based on hardware and desired accuracy (default is 20).
- **Learning Rate**: Initial learning rate set to 1e-4, adjusted automatically with `ReduceLROnPlateau`.

## Evaluation and Results

After training, the model's performance is evaluated on the validation set. 

Key metrics include:
- **Validation Accuracy**: Printed at the end of training.
- **Precision, Recall, F1-score**: Classification metrics generated in a report after predictions on the validation set.

Example output:
```
Validation Accuracy: 78.40%
              precision    recall  f1-score   support

           0       0.51      0.59      0.55       294
           1       0.51      0.43      0.47       294

    accuracy                           0.51       588
   macro avg       0.51      0.51      0.51       588
weighted avg       0.51      0.51      0.51       588
```

## Visualization

Training and validation accuracy/loss plots are generated to help track model performance over epochs. The plot is displayed automatically at the end of training and saved to the `plots` directory.

```python
import matplotlib.pyplot as plt

# Sample code snippet to plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy During Training')
plt.legend()
plt.show()
```

## Troubleshooting

- **Performance Warnings on M1/M2 Macs**: On Apple silicon Macs, TensorFlow may display a warning about optimizer performance. Switching to the legacy Adam optimizer (`tf.keras.optimizers.legacy.Adam`) is recommended.
- **Class Imbalance**: If the model accuracy is significantly imbalanced across classes, consider using class weights to balance the training.
- **Overfitting**: Increase dropout rate or apply early stopping if overfitting is observed.

## Acknowledgements

This project uses DenseNet121 pretrained on ImageNet, and draws inspiration from work in autism detection and image classification research.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This `README.md` provides a structured overview of the autism detection project, including instructions for setup, usage, and troubleshooting. Modify sections as needed to provide more specific information about your dataset, model, or experimental results.
