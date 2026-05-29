# ============================================================
# features/extractor.py
# DiagMAS-AI — Extraction de features
#
# Deux familles de features :
#   1. Features temporelles : RMS, crête, kurtosis, skewness, THD
#   2. Features MCSA (spectrales) : amplitude des bandes latérales
#      caractéristiques des défauts (§ I.3.5.1 du mémoire)
#
# Total : ~35 features par signal → vecteur X pour le modèle IA
# ============================================================

import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List
from configs.settings import MOTOR, FEAT_CFG, FeatureConfig


class FeatureExtractor:
    """
    Extrait un vecteur de features depuis un signal de courant statorique.
    """

    def __init__(self, fs: float = 10000.0, cfg: FeatureConfig = FEAT_CFG):
        self.fs  = fs
        self.dt  = 1.0 / fs
        self.cfg = cfg

    # ════════════════════════════════════════════════════════
    # 1. FEATURES TEMPORELLES
    # ════════════════════════════════════════════════════════

    def _temporal_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Indicateurs statistiques du signal temporel.
        Ces features sont sensibles aux défauts mécaniques et électriques.
        """
        feats = {}
        N = len(signal)

        # RMS (valeur efficace) — augmente avec les défauts
        rms = np.sqrt(np.mean(signal ** 2))
        feats["rms"] = rms

        # Valeur crête absolue
        peak = np.max(np.abs(signal))
        feats["peak"] = peak

        # Facteur de crête (crest factor) — sensible aux chocs
        feats["crest_factor"] = peak / (rms + 1e-10)

        # Kurtosis — détecte les impulsions (défauts ponctuels)
        feats["kurtosis"] = float(scipy_stats.kurtosis(signal, fisher=True))

        # Skewness — asymétrie de la distribution
        feats["skewness"] = float(scipy_stats.skew(signal))

        # Variance
        feats["variance"] = float(np.var(signal))

        # RMS des 10 dernières périodes (régime permanent estimé)
        n_steady = int(10 / MOTOR.fn * self.fs)  # 10 cycles
        steady   = signal[-n_steady:] if n_steady < N else signal
        feats["rms_steady"] = float(np.sqrt(np.mean(steady ** 2)))
        feats["std_steady"]  = float(np.std(steady))

        # Énergie normalisée
        feats["energy"] = float(np.sum(signal ** 2) / N)

        # Forme d'onde : rapport RMS à vide / en charge (approximé)
        half = N // 2
        rms1 = np.sqrt(np.mean(signal[:half] ** 2))
        rms2 = np.sqrt(np.mean(signal[half:] ** 2))
        feats["rms_ratio"] = float(rms2 / (rms1 + 1e-10))

        return feats

    # ════════════════════════════════════════════════════════
    # 2. FFT + FEATURES SPECTRALES (MCSA)
    # ════════════════════════════════════════════════════════

    def _compute_fft(self, signal: np.ndarray):
        """
        FFT avec fenêtre de Hann.
        Retourne (freqs, amplitudes_lineaires, amplitudes_dB).
        """
        N   = len(signal)
        win = np.hanning(N)
        fft_vals = np.fft.rfft(signal * win) / (N / 2)
        freqs    = np.fft.rfftfreq(N, d=self.dt)
        amp      = np.abs(fft_vals)
        amp_db   = 20 * np.log10(amp + 1e-12)
        return freqs, amp, amp_db

    def _sideband_amplitude(self, freqs: np.ndarray, amp: np.ndarray,
                             f_target: float, tol: float = None) -> float:
        """
        Amplitude linéaire de la bande latérale la plus proche de f_target.
        """
        tol = tol or self.cfg.sideband_tol_hz
        mask = (freqs >= f_target - tol) & (freqs <= f_target + tol)
        if not np.any(mask):
            return 0.0
        return float(np.max(amp[mask]))

    def _sideband_db(self, freqs: np.ndarray, amp: np.ndarray,
                      amp_db: np.ndarray, f_target: float,
                      f_fund: float) -> float:
        """
        Amplitude relative (dB) par rapport au fondamental.
        """
        amp_target = self._sideband_amplitude(freqs, amp, f_target)
        amp_ref    = self._sideband_amplitude(freqs, amp, f_fund, tol=1.5)
        if amp_ref < 1e-12:
            return -80.0
        return float(20 * np.log10((amp_target + 1e-12) / (amp_ref + 1e-12)))

    def _spectral_features(self, signal: np.ndarray, g: float = None) -> Dict[str, float]:
        """
        Features spectrales MCSA.
        g : glissement (si None, utilise glissement nominal).
        """
        g = g if g is not None else MOTOR.g_nom
        g = max(0.005, g)
        fn = MOTOR.fn                       # 50 Hz
        fr = fn * (1 - g) / MOTOR.p        # fréquence rotation [Hz]

        freqs, amp, amp_db = self._compute_fft(signal)
        feats = {}

        # ── Fondamental ────────────────────────────────────────
        mask_fund = (freqs >= fn - 2) & (freqs <= fn + 2)
        f_fund    = float(freqs[mask_fund][np.argmax(amp[mask_fund])]) if np.any(mask_fund) else fn
        amp_fund  = float(np.max(amp[mask_fund])) if np.any(mask_fund) else 1e-10
        feats["f_fundamental"] = f_fund
        feats["amp_fundamental"] = amp_fund

        # Harmoniques statoriques (3, 5, 7)
        for h in [3, 5, 7]:
            a = self._sideband_amplitude(freqs, amp, h * fn, tol=1.0)
            feats[f"amp_h{h}"] = a
            feats[f"thd_h{h}_db"] = 20 * np.log10((a + 1e-12) / (amp_fund + 1e-12))

        # THD global (§ I.3.5.1)
        thd = np.sqrt(sum(
            self._sideband_amplitude(freqs, amp, h * fn, tol=1.0) ** 2
            for h in [3, 5, 7]
        )) / (amp_fund + 1e-10)
        feats["thd"] = float(thd)

        # ── Signature barres cassées : (1 ± 2k·g)·fs (mémoire § II.6) ─
        for k in range(1, self.cfg.n_harmonics_barre + 1):
            for sign, label in [(+1, "sup"), (-1, "inf")]:
                f_barre = (1 + sign * 2 * k * g) * fn
                if f_barre > 0:
                    feats[f"barre_k{k}_{label}_amp"] = self._sideband_amplitude(
                        freqs, amp, f_barre)
                    feats[f"barre_k{k}_{label}_db"]  = self._sideband_db(
                        freqs, amp, amp_db, f_barre, fn)

        # ── Signature excentricité : fs ± m·fr (mémoire § II.7) ───────
        for m in range(1, self.cfg.n_harmonics_exc + 1):
            for sign, label in [(+1, "sup"), (-1, "inf")]:
                f_exc = fn + sign * m * fr
                if f_exc > 0:
                    feats[f"exc_m{m}_{label}_amp"] = self._sideband_amplitude(
                        freqs, amp, f_exc)
                    feats[f"exc_m{m}_{label}_db"]  = self._sideband_db(
                        freqs, amp, amp_db, f_exc, fn)

        # ── Bandes harmoniques barres haut : Nr·fr ± fs ────────────────
        f_bar_h_inf = MOTOR.Nr_bars * fr - fn
        f_bar_h_sup = MOTOR.Nr_bars * fr + fn
        feats["bar_harm_inf"] = self._sideband_amplitude(freqs, amp, f_bar_h_inf, tol=2.0)
        feats["bar_harm_sup"] = self._sideband_amplitude(freqs, amp, f_bar_h_sup, tol=2.0)

        # ── Énergie spectrale par bande de fréquence ───────────────────
        bands = [(0, 30), (30, 60), (60, 100), (100, 200)]
        for f_lo, f_hi in bands:
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            feats[f"energy_band_{f_lo}_{f_hi}"] = float(
                np.sum(amp[mask] ** 2)) if np.any(mask) else 0.0

        # ── Glissement estimé depuis FFT ───────────────────────────────
        # Cherche la bande (1-2g)·fs la plus forte
        g_candidates = np.linspace(0.01, 0.10, 50)
        best_g, best_amp = g, 0.0
        for g_c in g_candidates:
            fb = (1 - 2 * g_c) * fn
            a  = self._sideband_amplitude(freqs, amp, fb, tol=0.5)
            if a > best_amp:
                best_amp = a
                best_g   = g_c
        feats["estimated_slip"] = float(best_g)

        return feats

    # ════════════════════════════════════════════════════════
    # 3. VECTEUR FINAL
    # ════════════════════════════════════════════════════════

    def extract(self, signal: np.ndarray, g: float = None) -> np.ndarray:
        """
        Extrait toutes les features et retourne un vecteur numpy 1-D.
        """
        feats = {}
        feats.update(self._temporal_features(signal))
        feats.update(self._spectral_features(signal, g=g))
        # Tri alphabétique → ordre déterministe
        keys = sorted(feats.keys())
        return np.array([feats[k] for k in keys], dtype=np.float32)

    def feature_names(self, signal: np.ndarray, g: float = None) -> List[str]:
        """Retourne la liste ordonnée des noms de features."""
        feats = {}
        feats.update(self._temporal_features(signal))
        feats.update(self._spectral_features(signal, g=g))
        return sorted(feats.keys())

    def extract_batch(self, signals: np.ndarray, g: float = None,
                       verbose: bool = True) -> np.ndarray:
        """
        Extrait les features pour un batch de signaux.
        signals : (n_samples, N)
        Retourne : (n_samples, n_features)
        """
        results = []
        n = len(signals)
        for i, sig in enumerate(signals):
            results.append(self.extract(sig, g=g))
            if verbose and (i + 1) % 100 == 0:
                print(f"  Features : {i+1}/{n}", flush=True)
        return np.array(results, dtype=np.float32)
