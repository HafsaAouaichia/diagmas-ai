# ============================================================
# inference/predict.py
# DiagMAS-AI — Pipeline d'inférence production
#
# Usage en production :
#   from inference.predict import DiagnosticPipeline
#   pipeline = DiagnosticPipeline.load()
#   result = pipeline.diagnose(signal_array)
#
# Ou en ligne de commande :
#   python inference/predict.py --signal signal.npy
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from typing import Union, List, Dict

from configs.settings import MOTOR, DATA_CFG, MODELS_DIR, FAULT_CLASSES, CLASS_NAMES
from features.extractor import FeatureExtractor
from models.classifier import DiagMASModel


class DiagnosticPipeline:
    """
    Pipeline d'inférence production.

    Prend en entrée un signal de courant brut (ou un batch)
    et retourne le diagnostic complet.
    """

    def __init__(self, model: DiagMASModel = None, fs: float = None):
        self.model     = model
        self.fs        = fs or DATA_CFG.fs
        self.extractor = FeatureExtractor(fs=self.fs)
        self._loaded   = model is not None

    @classmethod
    def load(cls, model_path: str = None, fs: float = None) -> "DiagnosticPipeline":
        """
        Charge le pipeline depuis le modèle sauvegardé.
        """
        model = DiagMASModel.load(model_path)
        pipe  = cls(model=model, fs=fs)
        print(f"  Pipeline chargé — {len(CLASS_NAMES)} classes, "
              f"{len(model.feature_names)} features")
        return pipe

    # ── Diagnostic d'un signal ────────────────────────────────
    def diagnose(self, signal: np.ndarray,
                  g: float = None) -> Dict:
        """
        Diagnostique un signal de courant statorique.

        Paramètres
        ----------
        signal : np.ndarray (N,)
            Signal de courant brut [A], N = fs × durée
        g : float, optionnel
            Glissement estimé (si connu, améliore les features spectrales)

        Retourne
        --------
        dict avec clés :
          class_name, confidence, severity, severity_label,
          action, all_probas, features_used, timestamp
        """
        if not self._loaded:
            raise RuntimeError("Modèle non chargé. Appelez DiagnosticPipeline.load()")

        # Extraction features
        x = self.extractor.extract(signal, g=g)

        # Prédiction
        reports = self.model.predict_report(x.reshape(1, -1))
        result  = reports[0]

        # Enrichissement
        result["timestamp"]     = datetime.now().isoformat()
        result["signal_length"] = len(signal)
        result["fs_hz"]         = self.fs
        result["n_features"]    = len(x)

        return result

    # ── Batch ─────────────────────────────────────────────────
    def diagnose_batch(self, signals: np.ndarray,
                        g: float = None,
                        verbose: bool = True) -> List[Dict]:
        """
        Diagnostique un batch de signaux.
        signals : (n_samples, N)
        """
        X = self.extractor.extract_batch(signals, g=g, verbose=verbose)
        reports = self.model.predict_report(X)
        now = datetime.now().isoformat()
        for r in reports:
            r["timestamp"] = now
        return reports

    # ── Résumé console ────────────────────────────────────────
    @staticmethod
    def print_result(result: Dict):
        SEV_COLORS = {
            "Normal":   "\033[92m",   # vert
            "Attention":"\033[93m",   # jaune
            "Dégradé":  "\033[33m",   # orange
            "CRITIQUE": "\033[91m",   # rouge
        }
        RESET = "\033[0m"
        color = SEV_COLORS.get(result.get("severity_label", "Normal"), "")

        print("\n" + "═" * 55)
        print(f"  RÉSULTAT DIAGNOSTIC — DiagMAS-AI")
        print("═" * 55)
        print(f"  Défaut détecté  : {color}{result['class_name']}{RESET}")
        print(f"  Confiance       : {result['confidence']*100:.1f}%")
        print(f"  Sévérité        : {color}{result['severity']:.1f}% — {result['severity_label']}{RESET}")
        print(f"  Recommandation  : {result['action']}")
        print("\n  Probabilités par classe :")
        sorted_p = sorted(result["all_probas"].items(), key=lambda x: -x[1])
        for cname, prob in sorted_p:
            bar = "█" * int(prob * 30)
            print(f"    {cname:<25} {prob*100:5.1f}%  {bar}")
        print("═" * 55)


# ──────────────────────────────────────────────────────────────
# Exemple d'utilisation / test rapide
# ──────────────────────────────────────────────────────────────
def demo_inference(model_path: str = None):
    """
    Test rapide : génère un signal de chaque classe et diagnostique.
    Affiche les résultats pour vérifier que le modèle de production
    fonctionne correctement.
    """
    from data_engine.signal_generator import SignalGenerator, FaultParams
    import time

    print("\n" + "=" * 55)
    print("  DiagMAS-AI — DÉMO INFÉRENCE PRODUCTION")
    print("=" * 55)

    pipeline = DiagnosticPipeline.load(model_path)
    gen      = SignalGenerator(DATA_CFG, seed=999)

    test_cases = [
        ("sain",              FaultParams()),
        ("barre_4_adjacentes",FaultParams(n_broken=4, adjacent=True)),
        ("exc_60",            FaultParams(eccentricity=0.60)),
        ("mixte",             FaultParams(n_broken=4, adjacent=True, eccentricity=0.60)),
    ]

    results_summary = []
    for true_class, fp in test_cases:
        signal = gen.generate(fp)
        t0     = time.perf_counter()
        result = pipeline.diagnose(signal)
        ms     = (time.perf_counter() - t0) * 1000

        ok = "✓" if result["class_name"] == true_class else "✗"
        print(f"\n  {ok} Vrai : {true_class:<25} "
              f"Prédit : {result['class_name']:<25} "
              f"conf={result['confidence']*100:.0f}%  "
              f"sev={result['severity']:.0f}%  "
              f"({ms:.1f}ms)")
        results_summary.append({
            "true": true_class,
            "predicted": result["class_name"],
            "correct": result["class_name"] == true_class,
            "confidence": result["confidence"],
            "latency_ms": ms,
        })

    n_correct = sum(r["correct"] for r in results_summary)
    print(f"\n  Résultat démo : {n_correct}/{len(test_cases)} corrects")
    return results_summary


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DiagMAS-AI Inference")
    parser.add_argument("--signal",      type=str, help="Fichier .npy contenant le signal courant")
    parser.add_argument("--model-path",  type=str, help="Chemin du modèle .pkl")
    parser.add_argument("--fs",          type=float, default=10000.0)
    parser.add_argument("--demo",        action="store_true", help="Lance la démo")
    args = parser.parse_args()

    if args.demo:
        demo_inference(args.model_path)
    elif args.signal:
        signal = np.load(args.signal)
        pipeline = DiagnosticPipeline.load(args.model_path, fs=args.fs)
        result   = pipeline.diagnose(signal)
        DiagnosticPipeline.print_result(result)
        print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()
