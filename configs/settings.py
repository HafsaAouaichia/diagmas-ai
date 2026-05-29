# ============================================================
# configs/settings.py
# DiagMAS-AI — Configuration centrale
# Tous les paramètres du mémoire sont ici. Ne jamais mettre
# de constantes en dur dans le code : tout passe par ce fichier.
# ============================================================

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# ── Racine du projet ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
MODELS_DIR   = OUTPUTS_DIR / "models"
DATA_DIR     = OUTPUTS_DIR / "data"

for d in [OUTPUTS_DIR, MODELS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Paramètres machine (Tableau III.1 du mémoire) ────────────
@dataclass(frozen=True)
class MotorConfig:
    # Machine générale
    Pn:      float = 2200.0    # Puissance nominale [W]
    Vn:      float = 220.0     # Tension nominale [V] (phase-phase)
    Nn:      float = 1440.0    # Vitesse nominale [tr/min]
    fn:      float = 50.0      # Fréquence [Hz]
    p:       int   = 2         # Paires de pôles
    conn:    str   = "Y"       # Couplage
    L_act:   float = 0.097     # Longueur active [m]

    # Stator
    Ns_enc:  int   = 36        # Encoches statoriques
    D_s_int: float = 0.099     # Diamètre interne stator [m]
    D_s_ext: float = 0.173     # Diamètre externe stator [m]
    Rs:      float = 7.14      # Résistance statorique [Ω/phase]
    Ls:      float = 0.040     # Inductance propre statorique [H]
    Lls:     float = 0.004     # Inductance de fuite statorique [H]

    # Rotor
    Nr_bars: int   = 48        # Barres rotoriques
    D_r_ext: float = 0.09867   # Diamètre externe rotor [m]
    D_r_int: float = 0.034     # Diamètre interne rotor [m]
    gap:     float = 0.00033   # Entrefer [m]
    Rr:      float = 0.0035    # Résistance rotorique ramenée [Ω]
    Lr:      float = 0.042     # Inductance propre rotorique ramenée [H]
    Llr:     float = 0.003     # Inductance de fuite rotorique [H]
    Lm:      float = 0.038     # Inductance magnétisante [H]

    # Mécanique
    J:       float = 0.006     # Inertie [kg·m²]
    Bf:      float = 0.001     # Frottement [N·m·s]
    TL_nom:  float = 14.35     # Couple de charge nominal [N·m]

    # Résultats de simulation nominale (mémoire § II.5.2)
    I_saine:   float = 4.24    # Courant RMS régime permanent [A]
    Ir_barre:  float = 149.90  # Courant barre nominal [A]
    Te_nominal: float = 14.35  # Couple nominal [N·m]
    N_nominal:  float = 1421.0 # Vitesse nominale simulée [tr/min]

    # Dérivés
    @property
    def ns(self):     return 60 * self.fn / self.p          # [tr/min]
    @property
    def g_nom(self):  return (self.ns - self.Nn) / self.ns  # glissement nominal
    @property
    def ws(self):     return 2 * np.pi * self.fn            # [rad/s]
    @property
    def Vph(self):    return self.Vn / np.sqrt(3)           # tension de phase [V]
    @property
    def In(self):     return self.Pn / (np.sqrt(3) * self.Vn * 0.87)  # courant nominal [A]


# ── Classes de défauts (exactement celles du mémoire) ────────
FAULT_CLASSES = {
    0: "sain",
    1: "barre_1",           # 1 barre cassée
    2: "barre_1_par_pole",  # 1 barre cassée par pôle (§ II.6)
    3: "barre_2_opposees",  # 2 barres cassées par pôle opposé
    4: "barre_4_adjacentes",# 4 barres cassées adjacentes
    5: "exc_20",            # Excentricité 20%
    6: "exc_40",            # Excentricité 40%
    7: "exc_60",            # Excentricité 60%
    8: "mixte",             # Défaut mixte : 4 barres + 60% exc.
}
N_CLASSES = len(FAULT_CLASSES)
CLASS_NAMES = list(FAULT_CLASSES.values())

# ── Paramètres de génération de données ──────────────────────
@dataclass(frozen=True)
class DataConfig:
    fs:            float = 10000.0  # Fréquence d'échantillonnage [Hz]
    duration:      float = 2.0      # Durée de chaque signal [s]
    n_samples_per_class: int = 300  # Échantillons par classe
    noise_snr_db:  float = 30.0     # SNR du bruit blanc [dB]
    random_seed:   int   = 42

    @property
    def N(self):  return int(self.fs * self.duration)
    @property
    def dt(self): return 1.0 / self.fs


# ── Paramètres d'extraction de features ──────────────────────
@dataclass(frozen=True)
class FeatureConfig:
    # FFT
    fft_max_freq:    float = 200.0   # Fréquence max analysée [Hz]
    fft_window:      str   = "hann"  # Fenêtre : hann | hamming | blackman
    sideband_tol_hz: float = 0.8     # Tolérance extraction bande latérale [Hz]

    # Features temporelles
    use_rms:         bool = True
    use_crest:       bool = True
    use_kurtosis:    bool = True
    use_skewness:    bool = True
    use_thd:         bool = True     # Distorsion harmonique totale

    # Features MCSA (bandes latérales)
    n_harmonics_barre: int = 2       # k=1,2 pour f_barre = (1±2kg)fs
    n_harmonics_exc:   int = 2       # m=1,2 pour f_exc = fs ± m*fr


# ── Paramètres d'entraînement ─────────────────────────────────
@dataclass(frozen=True)
class TrainConfig:
    test_size:       float = 0.20    # 80/20 split
    val_size:        float = 0.10    # 10% validation (sur train)
    cv_folds:        int   = 5       # Cross-validation
    random_seed:     int   = 42

    # Random Forest (modèle principal)
    rf_n_estimators: int   = 300
    rf_max_depth:    int   = None    # None = profondeur max
    rf_n_jobs:       int   = -1      # Tous les cœurs

    # SVM (modèle de comparaison)
    svm_C:           float = 10.0
    svm_gamma:       str   = "scale"
    svm_kernel:      str   = "rbf"


# ── Instances globales (importées partout) ────────────────────
MOTOR   = MotorConfig()
DATA_CFG= DataConfig()
FEAT_CFG= FeatureConfig()
TRAIN_CFG= TrainConfig()
