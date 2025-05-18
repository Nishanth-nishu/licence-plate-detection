# Data Science Project for License Plate Detection and Recognition

## Overview
This project aims to develop a system for detecting and recognizing license plates using deep learning techniques. The project utilizes the YOLO (You Only Look Once) model for object detection and EasyOCR for optical character recognition.

## Project Structure
```
licence-plate-detection-project
|
├── src
│   ├── utils.py
│   ├── setup.py
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Data
- **Detection Images**: The `data/detection/test.zip` file contains images used for the license plate detection task.
- **Recognition Images**: The `data/recognition/Licplatesrecognition_train.zip` file contains images used for the license plate recognition task.
- **Annotations**: The `data/Licplatesdetection_train.csv` file contains annotations for the license plate detection images.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd data-science-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Extract the datasets:
   - Use the utility functions in `src/utils.py` to extract the zip files.

## Usage
- **Training**: Run the training script to train the YOLO model on the detection dataset.
  ```
  python src/train.py
  ```

- **Validation**: Validate the model's performance using the validation script.
  ```
  python src/validate.py
  ```

- **Testing**: Test the model on new images and visualize the results.
  ```
  python src/test.py
  ```

## Configuration
The project configuration, including dataset paths and model parameters, can be modified in the `src/config.yaml` file.

## Acknowledgments
This project utilizes the following libraries:
- PyTorch
- torchvision
- EasyOCR
- OpenCV
- Ultralytics YOLO

## License
This project is licensed under the MIT License.
