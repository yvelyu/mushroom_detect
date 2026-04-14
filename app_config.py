import os

MODEL_DIR = "model"
UPLOAD_DIR = "static/uploads"

for directory in [UPLOAD_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


IMAGE_SIZE = 224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


TOP_K = 5
CONFIDENCE_THRESHOLD = 0.3