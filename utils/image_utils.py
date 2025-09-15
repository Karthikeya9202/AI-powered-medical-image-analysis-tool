import cv2
import os

def preprocess_image(image_path, target_width=500):
    """Load & resize image while keeping aspect ratio."""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_height = int(target_width / aspect_ratio)
    resized = cv2.resize(image, (target_width, new_height))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized, rgb_image

def save_temp_file(uploaded_file):
    """Save uploaded file temporarily and return path."""
    ext = uploaded_file.name.split(".")[-1]
    path = f"temp_upload.{ext}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def cleanup_file(path):
    """Delete temporary file safely."""
    if os.path.exists(path):
        os.remove(path)
