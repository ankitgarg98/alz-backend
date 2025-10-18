# app.py
import io
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import SiameseCapsuleNetwork, load_model, load_references
from utils import transform, CLASS_NAMES

# ---------- Paths (robust to Azure's working dir) ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "my_backend" / "siamese_capsule_finetuned.pth"
REF_PATH   = BASE_DIR / "my_backend" / "reference_embeddings_vecs_finetuned.pt"
STATIC_DIR = BASE_DIR / "static"

# ---------- Device ----------
device = torch.device("cpu")

# (Optional) fewer CPU threads on shared plans
# torch.set_num_threads(1)

# ---------- Load artifacts ----------
# Print actual resolved paths so you can see them in Azure Log Stream if anything fails
print(f"[BOOT] Loading model from: {MODEL_PATH}")
print(f"[BOOT] Loading references from: {REF_PATH}")

model = load_model(str(MODEL_PATH), device)
reference_embeddings = load_references(str(REF_PATH), device, model)

# ---------- FastAPI ----------
app = FastAPI(title="Alzheimer MRI Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
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
            ds = [
                torch.nn.functional.pairwise_distance(
                    test_vec.unsqueeze(0),
                    r.unsqueeze(0),
                    p=2
                ).item()
                for r in refs
            ]
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
@app.get("/check-packages")
def check_packages():
    try:
        import uvicorn
        import fastapi
        return {
            "uvicorn": uvicorn.__version__,
            "fastapi": fastapi.__version__,
            "status": "success"
        }
    except ImportError as e:
        return {"status": "failed", "error": str(e)}

# ---------- Static (only if folder exists) ----------
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    print(f"[BOOT] Warning: static directory not found at {STATIC_DIR}. Skipping mount.")
