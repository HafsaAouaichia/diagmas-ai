#!/usr/bin/env python3
# ============================================================
# main.py — DiagMAS-AI
# Point d'entrée principal
#
# Usage :
#   python main.py train              # Entraînement complet
#   python main.py train --n 500      # 500 échantillons/classe
#   python main.py demo               # Démo inférence
#   python main.py api                # Lance l'API REST
#   python main.py test               # Lance les tests
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse


def cmd_train(args):
    from training.train import run_full_training
    run_full_training(
        n_per_class     = args.n,
        force_regen     = args.force_regen,
        force_reextract = args.force_reextract,
    )


def cmd_demo(args):
    from inference.predict import demo_inference
    demo_inference(args.model_path)


def cmd_api(args):
    try:
        import uvicorn
        uvicorn.run("api.app:app",
                    host=args.host, port=args.port, reload=args.reload)
    except ImportError:
        print("Installez FastAPI et uvicorn :")
        print("  pip install fastapi uvicorn")


def cmd_test(args):
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])


def cmd_info(args):
    from configs.settings import MOTOR, DATA_CFG, TRAIN_CFG, N_CLASSES, CLASS_NAMES
    print("\n" + "=" * 55)
    print("  DiagMAS-AI — Informations projet")
    print("=" * 55)
    print(f"\n  Machine : {MOTOR.Pn/1000:.1f} kW / {MOTOR.Vn} V / "
          f"{MOTOR.fn} Hz / {MOTOR.Nn} tr/min")
    print(f"  Pôles   : {MOTOR.p*2} | Barres rotor : {MOTOR.Nr_bars} | "
          f"Encoches : {MOTOR.Ns_enc}")
    print(f"\n  Classes ({N_CLASSES}) :")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    [{i}] {name}")
    print(f"\n  Données : {DATA_CFG.n_samples_per_class} échantillons/classe")
    print(f"  fs = {DATA_CFG.fs} Hz | durée = {DATA_CFG.duration} s")
    print(f"  SNR = {DATA_CFG.noise_snr_db} dB")
    print(f"\n  Entraînement : RF={TRAIN_CFG.rf_n_estimators} arbres | "
          f"SVM C={TRAIN_CFG.svm_C} | CV={TRAIN_CFG.cv_folds}-fold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiagMAS-AI — Diagnostic IA des défauts dans le moteur asynchrone"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Entraîner le modèle")
    p_train.add_argument("--n", type=int, default=300,
                          help="Échantillons par classe (défaut: 300)")
    p_train.add_argument("--force-regen",    action="store_true",
                          help="Régénère les données même si elles existent")
    p_train.add_argument("--force-reextract",action="store_true",
                          help="Ré-extrait les features même si elles existent")

    # demo
    p_demo = sub.add_parser("demo", help="Démo d'inférence")
    p_demo.add_argument("--model-path", type=str, default=None)

    # api
    p_api = sub.add_parser("api", help="Lance l'API REST")
    p_api.add_argument("--host",   default="0.0.0.0")
    p_api.add_argument("--port",   type=int, default=8000)
    p_api.add_argument("--reload", action="store_true")

    # test
    sub.add_parser("test", help="Lance les tests pytest")

    # info
    sub.add_parser("info", help="Affiche les infos du projet")

    args = parser.parse_args()
    {
        "train": cmd_train,
        "demo":  cmd_demo,
        "api":   cmd_api,
        "test":  cmd_test,
        "info":  cmd_info,
    }[args.cmd](args)
