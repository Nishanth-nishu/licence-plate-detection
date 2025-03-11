import os
import cv2
import torch
import yaml
from torchvision import transforms
from ultralytics import YOLO

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_model(model, val_images_dir, device):
    model.eval()  # Set the model to evaluation mode
    results = []

    # Get all image files in the validation directory
    image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(val_images_dir, image_file)
        image = cv2.imread(image_path)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model(image_tensor)
            results.append((image_file, result))

    return results

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)

    model = YOLO(config['model_path']).to(config['device'])
    val_images_dir = config['val_images_path']
    
    validation_results = validate_model(model, val_images_dir, config['device'])

    for image_file, result in validation_results:
        print(f"Validation results for {image_file}: {result}")