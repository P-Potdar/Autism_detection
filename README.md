Autism Detection using CNN with DenseNet121
This project aims to create a deep learning model to classify images for autism detection. Using transfer learning with a DenseNet121 backbone, the model is fine-tuned to identify patterns in images and classify them with binary labels (autism or non-autism).

Table of Contents
Project Overview
Project Structure
Dependencies
Installation
Usage
Model Training and Evaluation
Results
Troubleshooting
License
Project Overview
This project leverages the DenseNet121 model pretrained on ImageNet to perform binary classification. The dataset is preprocessed and augmented using Keras’ ImageDataGenerator to improve model generalization. The model incorporates data augmentation, custom dense layers, and learning rate scheduling to improve accuracy and manage class imbalance.

Key Features
Transfer Learning with DenseNet121.
Data Augmentation to enhance the dataset.
Fine-tuning of model layers to optimize performance.
Learning Rate Scheduling with ReduceLROnPlateau callback.
Project Structure
plaintext
Copy code
├── consolidated              # Directory containing train and validation datasets
│   ├── autism                # Folder with images labeled as autistic
│   ├── non_autism            # Folder with images labeled as non-autistic
├── cnn.py                    # Script to preprocess data, build and train model
├── requirements.txt          # Project dependencies
├── README.md                 # Project README
└── models                    # Directory where the trained model is saved
Dependencies
Python 3.8+
TensorFlow
NumPy
scikit-learn
Matplotlib
Please see requirements.txt for a full list of dependencies.

Installation
Clone the Repository

bash
Copy code
git clone https://github.com/your-username/autism-detection-cnn.git
cd autism-detection-cnn
Create a Virtual Environment

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
Install Required Packages

bash
Copy code
pip install -r requirements.txt
Usage
Data Preparation

Place your dataset images into consolidated/ with two subfolders:

autism/ for images with autism
non_autism/ for images without autism
Train the Model

Run the cnn.py script to start training:

bash
Copy code
python cnn.py
Evaluate the Model

The script outputs validation accuracy and a classification report after training completes. The trained model is saved in .h5 and .pkl formats.

Model Training and Evaluation
The model uses ImageDataGenerator for data augmentation.
Training and validation accuracy are visualized after each epoch.
ReduceLROnPlateau callback reduces the learning rate if validation accuracy plateaus.
Evaluation Metrics
The cnn.py script provides:

Validation Accuracy: Outputs the accuracy on the validation dataset.
Classification Report: Displays precision, recall, and F1-score for both classes.
Results
Metric	Value
Validation Accuracy	78.40%
Precision (Class 0)	0.51
Recall (Class 0)	0.59
Precision (Class 1)	0.51
Recall (Class 1)	0.43
F1-Score (Weighted)	0.51
Note: These metrics may vary based on dataset size and model training parameters.
