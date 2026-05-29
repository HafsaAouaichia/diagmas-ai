# ============================================================
# api/app.py
# DiagMAS-AI — API REST production (FastAPI)
#
# Endpoints :
#   POST /diagnose          → diagnostique 1 signal (JSON ou fichier)
#   POST /diagnose/batch    → diagnostique N signaux
#   GET  /health            → statut du serveur
#   GET  /classes           → liste des classes de défauts
#   GET  /model/info        → infos sur le modèle chargé
#
# Lancement :
#   uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from typing import List, Optional

# FastAPI est optionnel ; le fichier reste valide sans lui.
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from configs.settings import MOTOR, DATA_CFG, CLASS_NAMES, N_CLASSES, MODELS_DIR
from inference.predict import DiagnosticPipeline

# ── Chargement du modèle au démarrage ────────────────────────
_pipeline: DiagnosticPipeline = None

def get_pipeline() -> DiagnosticPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DiagnosticPipeline.load()
    return _pipeline


if HAS_FASTAPI:

    app = FastAPI(
        title       = "DiagMAS-AI",
        description = "Diagnostic des défauts dans le moteur asynchrone — API REST",
        version     = "1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["*"],
        allow_headers  = ["*"],
    )

    # ── Schémas ──────────────────────────────────────────────
    class SignalRequest(BaseModel):
        """Signal de courant statorique en JSON."""
        signal:      List[float] = Field(..., description="Courant phase A [A], N points")
        fs:          float       = Field(10000.0, description="Fréquence d'échantillonnage [Hz]")
        slip:        Optional[float] = Field(None, description="Glissement estimé (0..1)")
        motor_speed: Optional[float] = Field(None, description="Vitesse mesurée [tr/min]")

    class DiagnosticResponse(BaseModel):
        class_idx:     int
        class_name:    str
        confidence:    float
        severity:      float
        severity_label:str
        action:        str
        all_probas:    dict
        timestamp:     str
        signal_length: int
        fs_hz:         float

    class BatchSignalRequest(BaseModel):
        signals: List[List[float]] = Field(..., description="Liste de signaux")
        fs:      float = Field(10000.0)
        slip:    Optional[float] = None

    # ── Routes ───────────────────────────────────────────────
    @app.get("/health")
    def health():
        """Vérifie que l'API et le modèle sont opérationnels."""
        try:
            pipe = get_pipeline()
            return {
                "status":   "ok",
                "model":    "loaded",
                "classes":  N_CLASSES,
                "timestamp":datetime.now().isoformat(),
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/classes")
    def list_classes():
        """Liste des classes de défauts supportées."""
        return {
            "n_classes": N_CLASSES,
            "classes": {i: name for i, name in enumerate(CLASS_NAMES)},
        }

    @app.get("/model/info")
    def model_info():
        """Informations sur le modèle chargé."""
        pipe = get_pipeline()
        return {
            "model_type":   "Ensemble (RF + SVM + GradientBoost)",
            "n_features":   len(pipe.model.feature_names),
            "n_classes":    N_CLASSES,
            "class_names":  CLASS_NAMES,
            "motor": {
                "power_w":   MOTOR.Pn,
                "voltage_v": MOTOR.Vn,
                "speed_rpm": MOTOR.Nn,
                "freq_hz":   MOTOR.fn,
                "poles":     MOTOR.p * 2,
                "rotor_bars": MOTOR.Nr_bars,
            },
        }

    @app.post("/diagnose", response_model=DiagnosticResponse)
    def diagnose(req: SignalRequest):
        """
        Diagnostique un signal de courant statorique.

        Le signal doit contenir au moins fs × 0.5 points
        (au moins 0.5 seconde de signal).
        """
        sig = np.array(req.signal, dtype=np.float32)
        min_len = int(req.fs * 0.5)
        if len(sig) < min_len:
            raise HTTPException(
                status_code=422,
                detail=f"Signal trop court : {len(sig)} points (min {min_len})"
            )

        # Glissement depuis vitesse si fourni
        g = req.slip
        if g is None and req.motor_speed is not None:
            g = max(0.001, (MOTOR.ns - req.motor_speed) / MOTOR.ns)

        try:
            pipe   = get_pipeline()
            result = pipe.diagnose(sig, g=g)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return DiagnosticResponse(**result)

    @app.post("/diagnose/batch")
    def diagnose_batch(req: BatchSignalRequest):
        """Diagnostique un batch de signaux."""
        signals = np.array(req.signals, dtype=np.float32)
        if signals.ndim != 2:
            raise HTTPException(status_code=422,
                                detail="signals doit être une liste de listes (2D)")
        try:
            pipe    = get_pipeline()
            results = pipe.diagnose_batch(signals, g=req.slip, verbose=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"n_signals": len(results), "results": results}

    @app.post("/diagnose/file")
    async def diagnose_file(file: UploadFile = File(...),
                             fs: float = 10000.0,
                             slip: float = None):
        """
        Diagnostique depuis un fichier .npy uploadé.
        """
        if not file.filename.endswith(".npy"):
            raise HTTPException(status_code=422, detail="Fichier .npy requis")
        content = await file.read()
        import io
        sig = np.load(io.BytesIO(content))
        if sig.ndim != 1:
            raise HTTPException(status_code=422, detail="Signal 1-D requis")
        pipe   = get_pipeline()
        result = pipe.diagnose(sig.astype(np.float32), g=slip)
        return result


# ── Lancement direct ──────────────────────────────────────────
if __name__ == "__main__":
    if not HAS_FASTAPI:
        print("FastAPI non installé. Lancez : pip install fastapi uvicorn")
        print("L'API REST ne sera pas disponible.")
        print("Le reste du projet (training, inference) fonctionne sans FastAPI.")
    else:
        import uvicorn
        uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
