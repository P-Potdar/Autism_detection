# import os
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam

# # Paths for dataset
# data_dir = 'consolidated'  # Path to the dataset directory

# # Define the CNN model (DenseNet121 as base)
# def build_cnn_model():
#     base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
#     x = base_model.output
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(512, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     output = layers.Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification
#     model = models.Model(inputs=base_model.input, outputs=output)
#     return model

# # Build the CNN model
# cnn_model = build_cnn_model()

# # Compile the model
# cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# # ImageDataGenerator for data augmentation and validation
# datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # 20% for validation

# # Training generator
# train_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=(128, 128),  # Resize images to match the input size of the model
#     batch_size=32,
#     class_mode='binary',  # Binary classification (autism vs. non-autism)
#     subset='training',
#     shuffle=True  # Shuffle the training data
# )

# # Validation generator
# validation_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation',
#     shuffle=False  # No shuffle during validation for consistency
# )

# # Train the model
# cnn_model.fit(
#     train_generator,
#     epochs=10,  # Number of epochs
#     validation_data=validation_generator,
#     verbose=1
# )

# # Evaluate the model on the validation set
# loss, accuracy = cnn_model.evaluate(validation_generator, verbose=1)

# # Print the validation accuracy as a percentage
# print(f"Validation Accuracy: {accuracy * 100:.2f}%")
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Set dataset directory
data_dir = 'consolidated'

# Enhanced Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Zooming
    horizontal_flip=True,  # Horizontal flips
    fill_mode='nearest',  # Fill missing pixels after transformation
    validation_split=0.2  # Use 20% for validation
)

# Generators for training and validation sets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Training subset
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Validation subset
)

# Building the CNN Model (using DenseNet121 as base)
def build_cnn_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout to prevent overfitting
    output = Dense(1, activation='sigmoid')(x)
    
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[:100]:  # Freeze first 100 layers
        layer.trainable = False
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Compile and train the CNN model
cnn_model = build_cnn_model()
cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate reduction on plateau callback
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Training the model
history = cnn_model.fit(
    train_generator,
    epochs=20,  # Increase the epochs to give the model more time to learn
    validation_data=validation_generator,
    verbose=1,
    callbacks=[lr_scheduler]
)

# Model evaluation
val_accuracy = history.history['val_accuracy'][-1] * 100  # Get final validation accuracy
print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Optionally, plot training and validation accuracy to visualize progress
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy During Training')
plt.legend()
plt.show()

# Optionally, print out a classification report after predictions
from sklearn.metrics import classification_report

val_preds = cnn_model.predict(validation_generator, verbose=1)
val_preds = (val_preds > 0.5).astype(int)  # Convert to binary labels (0 or 1)

# Print classification report
print(classification_report(validation_generator.classes, val_preds))
