import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Load YOLOv8 model for license plate detection
model = YOLO("E:\Licence detection\license-plate-recognition\best (1).pt")  # Replace with your trained YOLOv8 weights

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def detect_and_recognize_license_plate(image):
    """
    Detects and recognizes license plates in an image.
    """
    # Convert Gradio image input to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect license plates using YOLOv8
    results = model(image)
    license_plates = []

    # Iterate through detected objects
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Crop the license plate region
            cropped_plate = image[y1:y2, x1:x2]

            # Use EasyOCR to extract text from the cropped plate
            ocr_results = reader.readtext(cropped_plate)
            plate_text = " ".join([res[1] for res in ocr_results])  # Combine all detected text

            # Draw bounding box and text on the original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Append detected text to the list
            license_plates.append(plate_text)

    # Convert the image back to RGB for Gradio display
    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Return the output image and detected license plate text
    return output_image, ", ".join(license_plates)

# Gradio interface
interface = gr.Interface(
    fn=detect_and_recognize_license_plate,
    inputs=gr.Image(label="Upload Image"),
    outputs=[gr.Image(label="Detected License Plate"), gr.Textbox(label="Extracted Text")],
    title="License Plate Detection and Recognition",
    description="Upload an image to detect and recognize license plates."
)

# Launch the app
interface.launch()