import os
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml

# Load configuration from YAML file
with open('src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the YOLO model
model = YOLO(config['model_path'])

# Function to run inference on a single image
def run_inference(image_path):
    image = cv2.imread(image_path)
    results = model(image)

    # Check if results are returned as a list and access the first image's results
    if isinstance(results, list):
        results = results[0]  # Get the first result from the list

    return results, image

# Function to visualize results
def visualize_results(results, image):
    for box in results.boxes:
        # Get bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()  # Confidence score
        cls = box.cls[0].item()  # Class ID

        # Draw bounding boxes on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image with bounding boxes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main function to test the model on a set of images
def main():
    test_images_dir = config['test_images_dir']
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in test_images:
        img_path = os.path.join(test_images_dir, img_file)
        results, image = run_inference(img_path)
        visualize_results(results, image)

if __name__ == "__main__":
    main()