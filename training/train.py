# ============================================================
# training/train.py
# DiagMAS-AI — Pipeline d'entraînement complet
#
# Étapes :
#   1. Génération des données synthétiques
#   2. Extraction des features
#   3. Train/Val/Test split
#   4. Entraînement des 3 modèles
#   5. Cross-validation
#   6. Évaluation complète (accuracy, F1, matrice de confusion)
#   7. Sauvegarde du meilleur modèle
#   8. Rapport de training
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from configs.settings import (
    MOTOR, DATA_CFG, FEAT_CFG, TRAIN_CFG,
    MODELS_DIR, DATA_DIR, OUTPUTS_DIR,
    FAULT_CLASSES, CLASS_NAMES, N_CLASSES
)
from data_engine.signal_generator import DatasetBuilder
from features.extractor import FeatureExtractor
from models.classifier import (
    DiagMASModel, build_random_forest,
    build_svm, build_gradient_boost, build_ensemble
)


# ──────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────
def _banner(text: str):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def _subsection(text: str):
    print(f"\n── {text} {'─' * (55 - len(text))}")


# ──────────────────────────────────────────────────────────────
# Étape 1 : Données
# ──────────────────────────────────────────────────────────────
def step_generate_data(n_per_class: int, force_regen: bool = False):
    raw_path = DATA_DIR / "raw_signals.npz"

    if raw_path.exists() and not force_regen:
        print(f"  Données existantes chargées depuis {raw_path}")
        signals, labels = DatasetBuilder.load_raw(str(raw_path))
    else:
        _subsection("Génération des signaux synthétiques")
        builder = DatasetBuilder()
        signals, labels = builder.build_raw(n_per_class=n_per_class)
        signals_arr = np.array(signals)
        labels_arr  = np.array(labels)
        np.savez_compressed(str(raw_path),
                            signals=signals_arr, labels=labels_arr)
        print(f"  Sauvé → {raw_path}")
        return signals_arr, labels_arr

    return signals, labels


# ──────────────────────────────────────────────────────────────
# Étape 2 : Features
# ──────────────────────────────────────────────────────────────
def step_extract_features(signals: np.ndarray, labels: np.ndarray,
                           force_reextract: bool = False):
    feat_path = DATA_DIR / "features.npz"

    if feat_path.exists() and not force_reextract:
        print(f"  Features existantes chargées depuis {feat_path}")
        data = np.load(str(feat_path))
        return data["X"], data["y"], list(data["feat_names"])

    _subsection("Extraction des features MCSA")
    extractor = FeatureExtractor(fs=DATA_CFG.fs)
    X = extractor.extract_batch(signals, verbose=True)
    y = labels.copy()

    # Noms des features (depuis 1 exemple)
    sample_signal = signals[0]
    feat_names = extractor.feature_names(sample_signal)
    print(f"  → {X.shape[1]} features par signal")

    np.savez_compressed(str(feat_path),
                        X=X, y=y,
                        feat_names=np.array(feat_names))
    print(f"  Sauvé → {feat_path}")
    return X, y, feat_names


# ──────────────────────────────────────────────────────────────
# Étape 3 : Split
# ──────────────────────────────────────────────────────────────
def step_split(X: np.ndarray, y: np.ndarray):
    _subsection("Train / Validation / Test split")

    # Test : 20%
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TRAIN_CFG.test_size,
        stratify=y, random_state=TRAIN_CFG.random_seed
    )
    # Val : 10% du total (~12.5% du trainval)
    val_frac = TRAIN_CFG.val_size / (1 - TRAIN_CFG.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac,
        stratify=y_trainval, random_state=TRAIN_CFG.random_seed
    )
    print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────────────────────
# Étape 4 : Entraînement + CV
# ──────────────────────────────────────────────────────────────
def step_train(X_train, y_train, X_val, y_val):
    _subsection("Entraînement des modèles")

    models = {
        "random_forest":   build_random_forest(),
        "svm":             build_svm(),
        "gradient_boost":  build_gradient_boost(),
    }
    trained = {}
    val_scores = {}

    for name, pipe in models.items():
        print(f"\n  [{name}] ...", end=" ", flush=True)
        pipe.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, pipe.predict(X_val))
        val_scores[name] = val_acc
        trained[name] = pipe
        print(f"val_acc = {val_acc * 100:.2f}%")

    # Ensemble (utilise les modèles déjà fittés individuellement)
    _subsection("Construction du modèle ensemble")
    from sklearn.ensemble import VotingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Le VotingClassifier a besoin d'être fitté sur les données scalées
    # On re-entraîne le scaler commun
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    rf_clf  = trained["random_forest"].named_steps["clf"]
    svm_clf = trained["svm"].named_steps["clf"]
    gb_clf  = trained["gradient_boost"].named_steps["clf"]

    # Re-fit les clf individuels sur données scalées (pour le voting)
    rf_ens  = build_random_forest().named_steps["clf"]
    svm_ens = build_svm().named_steps["clf"]
    gb_ens  = build_gradient_boost().named_steps["clf"]
    rf_ens.fit(X_train_sc,  y_train)
    svm_ens.fit(X_train_sc, y_train)
    gb_ens.fit(X_train_sc,  y_train)

    from sklearn.ensemble import VotingClassifier
    voting = VotingClassifier(
        estimators=[("rf", rf_ens), ("svm", svm_ens), ("gb", gb_ens)],
        voting="soft", weights=[0.4, 0.3, 0.3]
    )
    voting.estimators_ = [rf_ens, svm_ens, gb_ens]
    voting.le_         = None
    voting.classes_    = np.arange(N_CLASSES)

    # Wrap dans un pipeline avec scaler pré-fitté
    from sklearn.pipeline import Pipeline
    ensemble_pipe = Pipeline([("scaler", scaler), ("clf", voting)])
    # Le pipeline doit être fitté pour que predict fonctionne
    ensemble_pipe.fit(X_train, y_train)

    val_acc_ens = accuracy_score(y_val, ensemble_pipe.predict(X_val))
    trained["ensemble"] = ensemble_pipe
    val_scores["ensemble"] = val_acc_ens
    print(f"  [ensemble]       val_acc = {val_acc_ens * 100:.2f}%")

    return trained, val_scores


# ──────────────────────────────────────────────────────────────
# Étape 5 : Évaluation test
# ──────────────────────────────────────────────────────────────
def step_evaluate(trained: dict, X_test: np.ndarray, y_test: np.ndarray,
                   X_train: np.ndarray, y_train: np.ndarray) -> dict:
    _subsection("Évaluation finale sur le jeu de test")
    results = {}

    for name, pipe in trained.items():
        y_pred = pipe.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        f1_wei = f1_score(y_test, y_pred, average="weighted")
        cm     = confusion_matrix(y_test, y_pred)
        rep    = classification_report(y_test, y_pred,
                                       target_names=CLASS_NAMES,
                                       output_dict=True)

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wei,
            "confusion_matrix": cm.tolist(),
            "report": rep,
            "y_pred": y_pred,
        }

        print(f"  [{name:<18}] acc={acc*100:.2f}%  f1_macro={f1_mac*100:.2f}%")

    # Cross-validation sur le meilleur modèle (ensemble)
    _subsection("Cross-validation 5-fold (ensemble)")
    best_pipe = trained["ensemble"]
    X_all  = np.vstack([X_train, X_test])
    y_all  = np.concatenate([y_train, y_test])
    skf    = StratifiedKFold(n_splits=TRAIN_CFG.cv_folds,
                              shuffle=True,
                              random_state=TRAIN_CFG.random_seed)
    cv_acc = cross_val_score(best_pipe, X_all, y_all,
                              cv=skf, scoring="accuracy", n_jobs=-1)
    results["cross_val"] = {
        "scores": cv_acc.tolist(),
        "mean":   float(cv_acc.mean()),
        "std":    float(cv_acc.std()),
    }
    print(f"  CV acc = {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")

    return results


# ──────────────────────────────────────────────────────────────
# Étape 6 : Graphiques
# ──────────────────────────────────────────────────────────────
DARK = "#0a0d12"; BLUE = "#00d4ff"; ORANGE = "#ff6b35"
GREEN = "#39ff14"; YELLOW = "#ffbb00"; GRAY = "#5a7090"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor": "#1e2d45", "axes.labelcolor": GRAY,
    "xtick.color": GRAY, "ytick.color": GRAY,
    "text.color": "#c8d8e8", "grid.color": "#1e2d45",
    "font.size": 8, "font.family": "monospace",
})


def step_plot(results: dict, trained: dict,
              X_train: np.ndarray, feat_names: list,
              y_test: np.ndarray):
    out = OUTPUTS_DIR / "plots"
    out.mkdir(exist_ok=True)

    # ── 1. Matrices de confusion ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
    axes = axes.flatten()
    names_to_plot = ["random_forest", "svm", "gradient_boost", "ensemble"]

    for ax, name in zip(axes, names_to_plot):
        cm = np.array(results[name]["confusion_matrix"])
        ax.set_facecolor(DARK)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        ax.set_title(f"{name}  acc={results[name]['accuracy']*100:.1f}%",
                     color=BLUE, fontsize=8)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.5 else GRAY,
                        fontsize=7)
        plt.colorbar(im, ax=ax)

    fig.suptitle("Matrices de confusion — DiagMAS-AI", color=BLUE, fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out / "confusion_matrices.png"), dpi=150, facecolor=DARK)
    plt.close()

    # ── 2. Importance des features (Random Forest) ──────────
    rf_clf = trained["random_forest"].named_steps["clf"]
    importances = rf_clf.feature_importances_
    sorted_idx  = np.argsort(importances)[-20:]  # top 20

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK)
    ax.set_facecolor(DARK)
    colors = [GREEN if "barre" in feat_names[i]
              else ORANGE if "exc" in feat_names[i]
              else BLUE for i in sorted_idx]
    ax.barh(range(len(sorted_idx)),
            importances[sorted_idx],
            color=colors, edgecolor="#1e2d45")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=7)
    ax.set_title("Top 20 features (Random Forest) — importance MCSA",
                 color=BLUE, fontsize=9)
    ax.set_xlabel("Importance (Gini)", fontsize=8)
    ax.grid(axis="x", alpha=0.4)
    ax.spines[:].set_color("#1e2d45")

    # Légende
    from matplotlib.patches import Patch
    legend = [Patch(color=GREEN, label="Features barres"),
              Patch(color=ORANGE, label="Features excentricité"),
              Patch(color=BLUE, label="Autres features")]
    ax.legend(handles=legend, fontsize=7, facecolor="#0f1520",
              edgecolor="#1e2d45")

    plt.tight_layout()
    plt.savefig(str(out / "feature_importance.png"), dpi=150, facecolor=DARK)
    plt.close()

    # ── 3. Comparaison des modèles ──────────────────────────
    model_names = ["random_forest", "svm", "gradient_boost", "ensemble"]
    accuracies  = [results[n]["accuracy"] * 100 for n in model_names]
    f1_scores   = [results[n]["f1_macro"] * 100  for n in model_names]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK)
    ax.set_facecolor(DARK)
    x = np.arange(len(model_names))
    w = 0.35
    b1 = ax.bar(x - w/2, accuracies, w, color=BLUE, label="Accuracy (%)",
                edgecolor="#1e2d45")
    b2 = ax.bar(x + w/2, f1_scores,  w, color=ORANGE, label="F1 Macro (%)",
                edgecolor="#1e2d45")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Comparaison des modèles — DiagMAS-AI", color=BLUE, fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.spines[:].set_color("#1e2d45")
    ax.legend(fontsize=8, facecolor="#0f1520", edgecolor="#1e2d45")
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=7, color="#c8d8e8")
    plt.tight_layout()
    plt.savefig(str(out / "model_comparison.png"), dpi=150, facecolor=DARK)
    plt.close()

    print(f"  Graphiques → {out}")


# ──────────────────────────────────────────────────────────────
# Étape 7 : Rapport JSON
# ──────────────────────────────────────────────────────────────
def step_save_report(results: dict, val_scores: dict,
                      feat_names: list, n_per_class: int):
    report = {
        "timestamp":     datetime.now().isoformat(),
        "motor":         f"{MOTOR.Pn/1000:.1f}kW / {MOTOR.Vn}V / {MOTOR.fn}Hz / {MOTOR.Nn}tr/min",
        "n_classes":     N_CLASSES,
        "class_names":   CLASS_NAMES,
        "n_per_class":   n_per_class,
        "n_features":    len(feat_names),
        "feature_names": feat_names,
        "val_scores":    {k: round(v*100, 2) for k, v in val_scores.items()},
        "test_scores": {
            name: {
                "accuracy_pct":   round(res["accuracy"]*100, 2),
                "f1_macro_pct":   round(res["f1_macro"]*100, 2),
                "f1_weighted_pct":round(res["f1_weighted"]*100, 2),
            }
            for name, res in results.items()
            if name != "cross_val"
        },
        "cross_validation": {
            "folds":    TRAIN_CFG.cv_folds,
            "mean_pct": round(results["cross_val"]["mean"]*100, 2),
            "std_pct":  round(results["cross_val"]["std"]*100, 2),
            "scores":   [round(s*100, 2) for s in results["cross_val"]["scores"]],
        },
        "per_class_f1": {
            name: {
                cls: round(results[name]["report"].get(cls, {}).get("f1-score", 0)*100, 2)
                for cls in CLASS_NAMES
            }
            for name in ["random_forest", "svm", "gradient_boost", "ensemble"]
        }
    }

    path = OUTPUTS_DIR / "training_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Rapport JSON → {path}")
    return report


# ──────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────
def run_full_training(n_per_class: int = 300,
                       force_regen: bool = False,
                       force_reextract: bool = False):

    _banner("DiagMAS-AI — PIPELINE D'ENTRAÎNEMENT COMPLET")
    t0 = datetime.now()

    # 1. Données
    _banner("ÉTAPE 1/7 : Génération des données")
    signals, labels = step_generate_data(n_per_class, force_regen)
    print(f"  Total : {len(signals)} signaux × {signals.shape[1]} échantillons")

    # 2. Features
    _banner("ÉTAPE 2/7 : Extraction des features")
    X, y, feat_names = step_extract_features(signals, labels, force_reextract)
    print(f"  Dataset : X={X.shape}, y={y.shape}")

    # 3. Split
    _banner("ÉTAPE 3/7 : Split des données")
    X_train, X_val, X_test, y_train, y_val, y_test = step_split(X, y)

    # 4. Entraînement
    _banner("ÉTAPE 4/7 : Entraînement")
    trained, val_scores = step_train(X_train, y_train, X_val, y_val)

    # 5. Évaluation
    _banner("ÉTAPE 5/7 : Évaluation test")
    results = step_evaluate(trained, X_test, y_test, X_train, y_train)

    # 6. Graphiques
    _banner("ÉTAPE 6/7 : Graphiques")
    step_plot(results, trained, X_train, feat_names, y_test)

    # 7. Sauvegarde modèle production
    _banner("ÉTAPE 7/7 : Sauvegarde modèle de production")
    best_pipe = trained["ensemble"]
    prod_model = DiagMASModel(pipeline=best_pipe, feature_names=feat_names)
    model_path = prod_model.save()

    # Rapport
    report = step_save_report(results, val_scores, feat_names, n_per_class)

    elapsed = (datetime.now() - t0).total_seconds()
    _banner(f"ENTRAÎNEMENT TERMINÉ en {elapsed:.1f}s")
    print(f"\n  Meilleur modèle (ensemble) :")
    print(f"    Accuracy test    : {results['ensemble']['accuracy']*100:.2f}%")
    print(f"    F1 macro test    : {results['ensemble']['f1_macro']*100:.2f}%")
    print(f"    CV 5-fold        : {results['cross_val']['mean']*100:.2f}% ± {results['cross_val']['std']*100:.2f}%")
    print(f"\n  Modèle prêt pour production → {model_path}")

    return prod_model, results, report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DiagMAS-AI Training")
    parser.add_argument("--n-per-class", type=int, default=300)
    parser.add_argument("--force-regen",    action="store_true")
    parser.add_argument("--force-reextract",action="store_true")
    args = parser.parse_args()

    run_full_training(
        n_per_class=args.n_per_class,
        force_regen=args.force_regen,
        force_reextract=args.force_reextract,
    )
