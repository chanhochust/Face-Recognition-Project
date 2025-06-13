import os

# Create the model directory
model_dir = os.path.expanduser('~/.insightface/models/arcface')
os.makedirs(model_dir, exist_ok=True)

# Verify the file exists
if os.path.exists(os.path.expanduser('~/.insightface/models/arcface/w600k_r50.onnx')):
    print("Model downloaded successfully!")
else:
    print("Model download failed. Please provide the correct URL.")
