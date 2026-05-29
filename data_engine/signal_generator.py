# ============================================================
# data_engine/signal_generator.py
# DiagMAS-AI — Générateur de signaux physiques
#
# Génère des signaux de courant statorique réalistes
# pour chaque classe de défaut du mémoire.
# Basé sur le modèle d-q + modulation amplitude (MCSA).
#
# Physique utilisée (§ II.6 et II.7 du mémoire) :
#   Machine saine : i(t) = I * sin(2π·fs·t + φ) + bruit
#   Barres cassées: + oscillation à 2g·fs + harmoniques
#   Excentricité  : + bandes latérales à fs ± m·fr
#   Défaut mixte  : superposition des deux
# ============================================================

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from configs.settings import MOTOR, DataConfig, FAULT_CLASSES, DATA_CFG


@dataclass
class FaultParams:
    """Paramètres physiques d'un défaut — dérivés du mémoire."""
    # Barres cassées
    n_broken:    int   = 0        # Nombre de barres cassées
    adjacent:    bool  = True     # Adjacentes ou opposées
    # Excentricité
    eccentricity: float = 0.0     # 0..1
    # Courant RMS de référence (mémoire § II.5.2 : 4.24 A saine)
    I_ref:       float = MOTOR.I_saine
    # Glissement (variable selon charge)
    g:           float = MOTOR.g_nom   # ~0.04
    # Charge (facteur 0..1)
    load:        float = 1.0


class SignalGenerator:
    """
    Génère des signaux de courant statorique (phase A) synthétiques.

    Chaque signal reproduit les signatures physiques décrites dans
    le mémoire pour chaque type de défaut.
    """

    def __init__(self, cfg: DataConfig = DATA_CFG, seed: int = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.fs  = cfg.fs
        self.dt  = cfg.dt
        self.N   = cfg.N
        self.t   = np.arange(self.N) * self.dt

    # ── Génération d'un signal ────────────────────────────────
    def generate(self, fault_params: FaultParams) -> np.ndarray:
        """
        Retourne un vecteur numpy (N,) : courant phase A en Ampères.
        """
        fp = fault_params
        fn = MOTOR.fn                      # 50 Hz
        g  = fp.g * fp.load                # glissement effectif
        g  = max(0.005, g)                 # sécurité
        fr = fn * (1 - g) / MOTOR.p       # fréquence rotation [Hz]

        # ── Fondamental ────────────────────────────────────────
        phi_0 = self.rng.uniform(0, 2 * np.pi)
        I_amp = fp.I_ref * fp.load
        signal = I_amp * np.sin(2 * np.pi * fn * self.t + phi_0)

        # ── Harmoniques statoriques (3, 5, 7) ─────────────────
        # Toujours présentes, faible amplitude
        for h, amp_ratio in [(3, 0.03), (5, 0.015), (7, 0.008)]:
            phi_h = self.rng.uniform(0, 2 * np.pi)
            signal += I_amp * amp_ratio * np.sin(2 * np.pi * h * fn * self.t + phi_h)

        # ── Défaut barres cassées ──────────────────────────────
        if fp.n_broken > 0:
            # § II.6 : ondulation courant à 2g·fs
            # Amplitude proportionnelle au nombre de barres cassées
            frac = fp.n_broken / MOTOR.Nr_bars
            # Barres adjacentes → effet plus sévère (mémoire p.61)
            severity = frac * (1.4 if fp.adjacent else 0.9)

            # Modulation d'amplitude : i(t)·(1 + Am·cos(2π·2g·fs·t))
            f_mod = 2 * g * fn        # fréquence de modulation
            Am    = min(0.45, severity * 3.5)
            signal *= (1.0 + Am * np.cos(2 * np.pi * f_mod * self.t))

            # Bandes latérales dans le spectre : (1 ± 2k·g)·fs
            for k in [1, 2]:
                for sign in [+1, -1]:
                    fb = (1 + sign * 2 * k * g) * fn
                    if fb > 0:
                        amp_sb = I_amp * severity * 0.18 / k
                        phi_sb = self.rng.uniform(0, 2 * np.pi)
                        signal += amp_sb * np.sin(2 * np.pi * fb * self.t + phi_sb)

        # ── Défaut excentricité ────────────────────────────────
        if fp.eccentricity > 0:
            e = fp.eccentricity
            # § II.7 : bandes à fs ± m·fr
            for m in [1, 2]:
                for sign in [+1, -1]:
                    fe = fn + sign * m * fr
                    if fe > 0:
                        amp_e = I_amp * e * 0.12 / m
                        phi_e = self.rng.uniform(0, 2 * np.pi)
                        signal += amp_e * np.sin(2 * np.pi * fe * self.t + phi_e)

            # Légère modulation du fondamental
            Am_e = e * 0.08
            signal *= (1.0 + Am_e * np.sin(2 * np.pi * fr * self.t))

        # ── Bruit blanc additif ────────────────────────────────
        snr_linear = 10 ** (self.cfg.noise_snr_db / 20)
        signal_power = np.sqrt(np.mean(signal ** 2))
        noise_amp    = signal_power / snr_linear
        signal += self.rng.normal(0, noise_amp, size=self.N)
        signal = ajouter_bruit_industriel(signal, fs=self.fs, intensite=1.0)


        return signal.astype(np.float32)

    # ── Paramètres par classe (calibrés sur résultats mémoire) ─
    @staticmethod
    def params_for_class(class_name: str, rng: np.random.Generator) -> FaultParams:
        """
        Retourne des FaultParams pour une classe donnée.
        La charge et le glissement sont légèrement variés pour
        augmenter la diversité du dataset d'entraînement.
        """
        load = rng.uniform(0.85, 1.05)   # ±15% charge nominale
        g    = MOTOR.g_nom * rng.uniform(0.8, 1.2)

        base = FaultParams(load=load, g=g)

        mapping = {
            "sain":             FaultParams(load=load, g=g),
            "barre_1":          FaultParams(n_broken=1, adjacent=True,  load=load, g=g),
            "barre_1_par_pole": FaultParams(n_broken=2, adjacent=False, load=load, g=g),
            "barre_2_opposees": FaultParams(n_broken=4, adjacent=False, load=load, g=g),
            "barre_4_adjacentes":FaultParams(n_broken=4, adjacent=True,  load=load, g=g),
            "exc_20":           FaultParams(eccentricity=0.20,           load=load, g=g),
            "exc_40":           FaultParams(eccentricity=0.40,           load=load, g=g),
            "exc_60":           FaultParams(eccentricity=0.60,           load=load, g=g),
            "mixte":            FaultParams(n_broken=4, adjacent=True,
                                            eccentricity=0.60,           load=load, g=g),
        }
        return mapping.get(class_name, base)


# ── Générateur de dataset complet ────────────────────────────
class DatasetBuilder:
    """
    Construit le dataset complet (X, y) prêt pour l'entraînement.
    """

    def __init__(self, cfg: DataConfig = DATA_CFG):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

    def build_raw(self, n_per_class: int = None) -> Tuple[list, list]:
        """
        Retourne (signals, labels) :
          signals : liste de np.ndarray (N,)
          labels  : liste d'int (index de classe)
        """
        if n_per_class is None:
            n_per_class = self.cfg.n_samples_per_class

        gen = SignalGenerator(self.cfg, seed=self.cfg.random_seed)
        signals, labels = [], []

        for class_idx, class_name in FAULT_CLASSES.items():
            print(f"  Génération classe [{class_idx}] {class_name:25s} × {n_per_class}", flush=True)
            for i in range(n_per_class):
                # Chaque signal a son propre seed pour la reproductibilité
                local_rng = np.random.default_rng(self.cfg.random_seed + class_idx * 10000 + i)
                gen.rng = local_rng
                fp = SignalGenerator.params_for_class(class_name, local_rng)
                sig = gen.generate(fp)
                signals.append(sig)
                labels.append(class_idx)

        return signals, labels

    def save_raw(self, signals: list, labels: list, path=None):
        import numpy as np
        path = path or (DATA_CFG.__class__.__name__ and
                        str(DATA_DIR := __import__('configs.settings',
                        fromlist=['DATA_DIR']).DATA_DIR / "raw_signals.npz"))
        from configs.settings import DATA_DIR
        path = path or str(DATA_DIR / "raw_signals.npz")
        np.savez_compressed(path,
                            signals=np.array(signals),
                            labels=np.array(labels))
        print(f"  Sauvé → {path}")
        return path
    


    @staticmethod
    def load_raw(path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        return data["signals"], data["labels"]
    

def ajouter_bruit_industriel(signal, fs=10000, intensite=1.0):
    """
    Simule le bruit d'une vraie usine algérienne.
    intensite : 0.5 = bruit léger, 1.0 = normal, 2.0 = usine très bruyante
    """
    rng = np.random.default_rng()
    
    # 1. Bruit thermique / électronique de mesure
    signal = signal + rng.normal(0, 0.04 * intensite * np.std(signal), len(signal))
    
    # 2. Harmoniques réseau parasites (autres machines dans l'usine)
    t = np.arange(len(signal)) / fs
    signal += 0.03 * intensite * np.sin(2 * np.pi * 150 * t + rng.uniform(0, np.pi))
    signal += 0.02 * intensite * np.sin(2 * np.pi * 250 * t + rng.uniform(0, np.pi))
    signal += 0.01 * intensite * np.sin(2 * np.pi * 350 * t + rng.uniform(0, np.pi))
    
    # 3. Impulsions mécaniques (chocs, vibrations de la structure)
    n_impulsions = int(15 * intensite)
    idx = rng.choice(len(signal), size=n_impulsions, replace=False)
    signal[idx] += rng.normal(0, 0.25 * intensite * np.std(signal), n_impulsions)
    
    # 4. Dérive lente (variation de charge dans le temps)
    derive = 0.02 * intensite * np.sin(2 * np.pi * 0.5 * t)
    signal = signal * (1 + derive)
    
    return signal
