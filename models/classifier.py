# ============================================================
# models/classifier.py
# DiagMAS-AI — Modèles de classification
#
# Trois modèles comparés (pratique recommandée en production) :
#   1. Random Forest   — robuste, interprétable, rapide
#   2. SVM RBF         — référence classique MCSA
#   3. Gradient Boost  — meilleure précision
#   4. Ensemble vote   — combinaison des trois (production)
# ============================================================

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from configs.settings import MOTOR, TRAIN_CFG, MODELS_DIR, N_CLASSES, CLASS_NAMES


# ── Construit un pipeline complet (scaler + modèle) ──────────
def build_random_forest() -> Pipeline:
    """
    Random Forest — modèle principal.
    Très bon sur données MCSA (Bellini et al., IEEE Trans. 2008 [06]).
    """
    clf = RandomForestClassifier(
        n_estimators   = TRAIN_CFG.rf_n_estimators,
        max_depth       = TRAIN_CFG.rf_max_depth,
        min_samples_split = 3,
        min_samples_leaf  = 1,
        max_features    = "sqrt",
        class_weight    = "balanced",
        random_state    = TRAIN_CFG.random_seed,
        n_jobs          = TRAIN_CFG.rf_n_jobs,
        oob_score       = True,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf)
    ])


def build_svm() -> Pipeline:
    """SVM RBF — référence en littérature MCSA."""
    clf = SVC(
        C           = TRAIN_CFG.svm_C,
        gamma       = TRAIN_CFG.svm_gamma,
        kernel      = TRAIN_CFG.svm_kernel,
        probability = True,           # Pour predict_proba
        class_weight= "balanced",
        random_state= TRAIN_CFG.random_seed,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf)
    ])


def build_gradient_boost() -> Pipeline:
    """Gradient Boosting — meilleure précision sur features MCSA."""
    clf = GradientBoostingClassifier(
        n_estimators    = 200,
        learning_rate   = 0.08,
        max_depth       = 5,
        subsample       = 0.85,
        min_samples_leaf= 2,
        random_state    = TRAIN_CFG.random_seed,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf)
    ])


def build_ensemble(rf, svm, gb) -> Pipeline:
    """
    Modèle de production : vote pondéré (soft voting).
    Combine RF, SVM et GB → meilleure généralisation.
    """
    ensemble = VotingClassifier(
        estimators=[
            ("rf",  rf.named_steps["clf"]),
            ("svm", svm.named_steps["clf"]),
            ("gb",  gb.named_steps["clf"]),
        ],
        voting = "soft",
        weights= [0.4, 0.3, 0.3],
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    ensemble)
    ])


# ── Wrapper de production ─────────────────────────────────────
class DiagMASModel:
    """
    Interface de production unique.
    Charge / sauvegarde le modèle, et expose predict() + predict_proba().
    """

    MODEL_FILENAME = "diagmas_model.pkl"

    def __init__(self, pipeline: Pipeline = None, feature_names: list = None):
        self.pipeline      = pipeline
        self.feature_names = feature_names or []
        self.class_names   = CLASS_NAMES
        self.n_classes     = N_CLASSES

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X : (n_samples, n_features) ou (n_features,)
        Retourne : array d'indices de classes (int)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités de chaque classe.
        X : (n_samples, n_features)
        Retourne : (n_samples, n_classes)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict_proba(X)

    def predict_named(self, X: np.ndarray) -> list:
        """Retourne les noms de classes (str) au lieu des indices."""
        indices = self.predict(X)
        return [self.class_names[i] for i in indices]

    def predict_report(self, X: np.ndarray) -> list:
        """
        Retourne pour chaque échantillon un dict complet :
        {
          "class_idx": int,
          "class_name": str,
          "confidence": float,    # proba de la classe prédite
          "all_probas": dict,     # {class_name: proba}
          "severity": float,      # 0..100
          "action": str,          # recommandation
        }
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        indices = self.predict(X)
        probas  = self.predict_proba(X)
        results = []

        for idx, prob_vec in zip(indices, probas):
            class_name  = self.class_names[idx]
            confidence  = float(prob_vec[idx])
            all_p       = {self.class_names[i]: float(p) for i, p in enumerate(prob_vec)}
            severity    = self._severity_score(class_name, confidence)
            action      = self._recommend_action(class_name, severity)

            results.append({
                "class_idx":   int(idx),
                "class_name":  class_name,
                "confidence":  round(confidence, 4),
                "all_probas":  all_p,
                "severity":    round(severity, 1),
                "severity_label": _severity_label(severity),
                "action":      action,
            })
        return results

    @staticmethod
    def _severity_score(class_name: str, confidence: float) -> float:
        """
        Score de sévérité 0..100 basé sur la classe et la confiance.
        Calibré d'après les résultats du mémoire (§ II.6, II.7).
        """
        base = {
            "sain":             0,
            "barre_1":          25,
            "barre_1_par_pole": 35,
            "barre_2_opposees": 50,
            "barre_4_adjacentes": 65,
            "exc_20":           20,
            "exc_40":           40,
            "exc_60":           65,
            "mixte":            85,
        }.get(class_name, 0)
        return base * confidence

    @staticmethod
    def _recommend_action(class_name: str, severity: float) -> str:
        if class_name == "sain":
            return "Aucune intervention requise. Surveillance normale."
        if severity < 20:
            return "Surveiller — augmenter la fréquence d'inspection."
        if severity < 45:
            return "Planifier une maintenance dans les 30 jours."
        if severity < 70:
            return "Maintenance préventive dans les 7 jours."
        return "ARRÊT RECOMMANDÉ — intervention immédiate."

    # ── Persistance ───────────────────────────────────────────
    def save(self, path: str = None) -> str:
        path = path or str(MODELS_DIR / self.MODEL_FILENAME)
        payload = {
            "pipeline":      self.pipeline,
            "feature_names": self.feature_names,
            "class_names":   self.class_names,
        }
        joblib.dump(payload, path, compress=3)
        print(f"  Modèle sauvé → {path}")
        return path

    @classmethod
    def load(cls, path: str = None) -> "DiagMASModel":
        path = path or str(MODELS_DIR / cls.MODEL_FILENAME)
        payload = joblib.load(path)
        model = cls(
            pipeline      = payload["pipeline"],
            feature_names = payload["feature_names"],
        )
        model.class_names = payload.get("class_names", CLASS_NAMES)
        return model


def _severity_label(sev: float) -> str:
    if sev < 20:  return "Normal"
    if sev < 45:  return "Attention"
    if sev < 70:  return "Dégradé"
    return "CRITIQUE"
