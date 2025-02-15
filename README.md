# AI-Powered Face Mask Detector

## Overview
The **AI-Powered Face Mask Detector** is a deep learning-based system that detects whether a person is wearing a mask or not in real-time. This project is useful for enforcing mask-wearing policies in public places, workplaces, and healthcare environments.

## Features
- Real-time face mask detection
- Uses deep learning and OpenCV for image processing
- Supports webcam and image-based detection
- Lightweight and efficient model

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Namith2002/AI-Powered_Face_Mask_Detector.git
   cd AI-Powered_Face_Mask_Detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model (if applicable) and place it in the appropriate directory.

## Usage
### Run the Face Mask Detector
To run the model on a webcam:
```bash
python detect_mask_webcam.py
```

To run on an image:
```bash
python detect_mask_image.py --image path/to/image.jpg
```

## Model Training
If you want to train the model from scratch, use the following command:
```bash
python train_mask_detector.py
```

## Dataset
This project uses a dataset of labeled images with and without masks. You can expand the dataset with more images for better accuracy.

## Results
- Achieves high accuracy in detecting masks
- Can work in real-time with minimal latency

## Contribution
Feel free to contribute by:
- Adding more features
- Improving the accuracy of the model
- Enhancing real-time detection

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contact
For any queries or contributions, reach out to:
- **Your Name**
- Email: your.email@example.com
- GitHub: [yourusername](https://github.com/yourusername)

