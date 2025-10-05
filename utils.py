# utils.py
from torchvision import transforms
from PIL import Image

# -----------------------------
# Class names
# -----------------------------
CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

# -----------------------------
# Resize with padding
# -----------------------------
class ResizeWithPadding:
    def __init__(self, target_size):
        self.target_size = target_size  # (H, W)

    def __call__(self, image: Image.Image):
        W, H = image.size
        r = min(self.target_size[1] / W, self.target_size[0] / H)
        newW, newH = int(W * r), int(H * r)
        image = image.resize((newW, newH), Image.Resampling.LANCZOS)
        canvas = Image.new("L", self.target_size, color=0)
        canvas.paste(image, ((self.target_size[1] - newW) // 2,
                             (self.target_size[0] - newH) // 2))
        return canvas

# -----------------------------
# Transform pipeline
# -----------------------------
transform = transforms.Compose([
    ResizeWithPadding((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
