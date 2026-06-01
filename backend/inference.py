import os
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "trained_models",
    "vit_ecg_model"
)

print("MODEL PATH:", MODEL_PATH)

processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

model = ViTForImageClassification.from_pretrained(MODEL_PATH)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

label_map = {
    0: "Normal ECG",
    1: "Abnormal Heartbeat",
    2: "Myocardial Infarction",
    3: "Post MI History"
}

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = outputs.logits.argmax(-1).item()

    import torch.nn.functional as F

    probs = F.softmax(outputs.logits, dim=1)
    confidence = probs.max().item() * 100

    return label_map[prediction], confidence