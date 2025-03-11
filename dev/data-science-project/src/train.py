import os
import zipfile
import pandas as pd
import shutil
import random
import yaml
from ultralytics import YOLO

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def prepare_data(images_directory, labels_directory, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.8):
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_directory) if f.endswith('.jpg')]
    random.shuffle(image_files)

    train_size = int(len(image_files) * split_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    move_files(train_files, images_directory, labels_directory, train_images_dir, train_labels_dir)
    move_files(val_files, images_directory, labels_directory, val_images_dir, val_labels_dir)

def move_files(files, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
    for file in files:
        shutil.move(os.path.join(src_image_dir, file), os.path.join(dest_image_dir, file))
        label_file = os.path.splitext(file)[0] + '.txt'
        shutil.move(os.path.join(src_label_dir, label_file), os.path.join(dest_label_dir, label_file))

def create_yaml(train_images_path, val_images_path, num_classes=1, class_names=['license_plate']):
    data_yaml = {
        'train': train_images_path,
        'val': val_images_path,
        'nc': num_classes,
        'names': class_names
    }

    yaml_file_path = 'data.yaml'
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data_yaml, yaml_file)

def train_model(data_yaml_path, epochs=50, batch_size=16, img_size=640):
    model = YOLO('yolov8n.yaml')
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project='/content/yolo_training',
        name='license_plate_model',
        save=True,
        exist_ok=True,
        device=0,
        augment=True
    )
    return results

if __name__ == "__main__":
    # Paths
    detection_zip_path = '../data/detection/test.zip'
    recognition_zip_path = '../data/recognition/Licplatesrecognition_train.zip'
    csv_file_path = "../data/Licplatesdetection_train.csv"
    images_directory = "../data/extracted_detection_images/license_plates_detection_train"
    labels_directory = "../data/yolo_labels"
    train_images_dir = "../data/train/images"
    train_labels_dir = "../data/train/labels"
    val_images_dir = "../data/val/images"
    val_labels_dir = "../data/val/labels"

    # Extract zip files
    extract_zip(detection_zip_path, images_directory)
    extract_zip(recognition_zip_path, labels_directory)

    # Prepare data
    prepare_data(images_directory, labels_directory, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir)

    # Create YAML file
    create_yaml(train_images_dir, val_images_dir)

    # Train model
    train_model('data.yaml')