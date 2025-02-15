import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("mask_detector_model.h5")
IMG_SIZE = 128

# Define labels
LABELS = ["No Mask", "Mask"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    label = LABELS[int(prediction[0] > 0.5)]
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    # Display result
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Face Mask Detector", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
