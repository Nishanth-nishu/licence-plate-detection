def extract_zip(zip_path, extract_to):
    """Extracts a zip file to the specified directory."""
    import os
    import zipfile

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def load_csv(csv_file_path):
    """Loads a CSV file and returns a DataFrame."""
    import pandas as pd

    data = pd.read_csv(csv_file_path)
    data.columns = [col.strip() for col in data.columns]  # Clean column names
    return data

def create_directory(directory):
    """Creates a directory if it doesn't exist."""
    import os

    os.makedirs(directory, exist_ok=True)

def move_files(files, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
    """Moves image and label files to the specified directories."""
    import shutil

    for file in files:
        # Move images
        shutil.move(os.path.join(src_image_dir, file), os.path.join(dest_image_dir, file))

        # Move corresponding label files
        label_file = os.path.splitext(file)[0] + '.txt'  # Assuming labels have the same name as images
        shutil.move(os.path.join(src_label_dir, label_file), os.path.join(dest_label_dir, label_file))