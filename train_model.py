import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

# Directories
DATASET_DIR = "dataset/"  # Update this path to your dataset directory
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = 128

# Load data
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)
    class_label = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(img_resized)
            labels.append(class_label)
        except Exception as e:
            print(f"Error loading image: {img} - {e}")

# Normalize and convert to NumPy arrays
data = np.array(data) / 255.0
labels = np.array(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
train_datagen.fit(x_train)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_datagen.flow(x_train, y_train, batch_size=32),
                    validation_data=(x_test, y_test), epochs=10)

# Save the trained model
model.save("mask_detector_model.h5")
print("Model saved as mask_detector_model.h5")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
