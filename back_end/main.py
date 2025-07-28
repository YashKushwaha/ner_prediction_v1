import os
from pathlib import Path

from fastapi import FastAPI

from src.inference_pipeline import NERModel

from back_end.routes import ui_routes, ner_routes
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "local_only", "checkpoints")
ner_model = NERModel(checkpoint_dir=CHECKPOINT_DIR)

app = FastAPI()

app.include_router(ui_routes.router)
app.include_router(ner_routes.router)

app.state.ner_model = ner_model


if __name__ == "__main__":
    import uvicorn
    app_path = Path(__file__).resolve().with_suffix('').name  # gets filename without .py
    uvicorn.run(f"{app_path}:app", host="localhost", port=8000, reload=True, workers = 4)