# app.py
import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import SiameseCapsuleNetwork, load_model, load_references
from utils import transform, CLASS_NAMES

device = torch.device("cpu")
MODEL_PATH = "my_backend/siamese_capsule_finetuned.pth"
REF_PATH   = "my_backend/reference_embeddings_vecs_finetuned.pt"

# Load artifacts on import (ok for your use-case)
model = load_model(MODEL_PATH, device)
reference_embeddings = load_references(REF_PATH, device, model)

app = FastAPI(title="Alzheimer MRI Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("L")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        test_vec = model.capsule_net(x).squeeze(0)

        d_means = []
        for cls in range(4):
            refs = reference_embeddings[cls]
            ds = [torch.nn.functional.pairwise_distance(
                    test_vec.unsqueeze(0),
                    r.unsqueeze(0),
                    p=2
                  ).item() for r in refs]
            d_means.append(sum(ds) / len(ds))

        inv = 1.0 / (torch.tensor(d_means) + 1e-8)
        close = (100.0 * (inv / inv.sum())).tolist()
        pred = int(torch.argmin(torch.tensor(d_means)))

        return {
            "prediction": CLASS_NAMES[pred],
            "closeness": [
                {"class": c, "score": round(close[i], 2)} for i, c in enumerate(CLASS_NAMES)
            ]
        }

# ✅ Mount SPA last so it doesn't shadow /predict or /docs
app.mount("/", StaticFiles(directory="static", html=True), name="static")
