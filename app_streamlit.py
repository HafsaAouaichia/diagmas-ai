# """
# app_streamlit.py — DiagMAS-AI
# Dashboard professionnel de diagnostic des défauts du moteur asynchrone.

# CORRECTION CRITIQUE par rapport à la version précédente :
#   - La démo utilise le VRAI générateur de signaux physiques du projet
#     (data_engine.signal_generator) et non des signaux simplifiés ad-hoc.
#   - L'upload CSV utilise le vrai extracteur de features.
#   - Les confidences affichées sont donc cohérentes avec la précision
#     réelle du modèle (97%+).

# Lancement :
#     streamlit run app_streamlit.py
# """

# import sys, os
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent))

# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import joblib
# import time

# # ─── Config page ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="DiagMAS-AI — Diagnostic Moteur Asynchrone",
#     page_icon="⚙️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─── CSS ─────────────────────────────────────────────────────
# st.markdown("""
# <style>
# /* Fond global sombre */
# .stApp { background-color: #0a0d12; color: #c8d8e8; }
# section[data-testid="stSidebar"] { background-color: #0f1520; border-right: 1px solid #1e2d45; }

# /* Header principal */
# .main-header {
#     background: linear-gradient(135deg, #001833 0%, #003d5c 100%);
#     padding: 1.5rem 2rem; border-radius: 12px;
#     margin-bottom: 1.5rem;
#     border: 1px solid #00d4ff44;
#     box-shadow: 0 4px 32px rgba(0,212,255,0.1);
# }
# .main-header h1 { color: #00d4ff; margin: 0; font-size: 1.8rem; letter-spacing: 2px; }
# .main-header p  { color: #5a7090; margin: 0.4rem 0 0 0; font-size: 0.9rem; letter-spacing: 1px; }

# /* Cartes KPI */
# .kpi-card {
#     background: #0f1520; border: 1px solid #1e2d45;
#     border-radius: 10px; padding: 1rem 1.2rem;
#     text-align: center;
# }
# .kpi-title { font-size: 0.75rem; color: #5a7090; letter-spacing: 2px; text-transform: uppercase; }
# .kpi-value { font-size: 2rem; font-weight: 700; font-family: monospace; }
# .kpi-unit  { font-size: 0.75rem; color: #5a7090; }

# /* Résultat diagnostic */
# .result-ok   { background: linear-gradient(135deg,#003d1a,#005c28); border:1px solid #39ff14; border-radius:12px; padding:1.5rem; text-align:center; }
# .result-warn { background: linear-gradient(135deg,#3d2e00,#5c4400); border:1px solid #ffbb00; border-radius:12px; padding:1.5rem; text-align:center; }
# .result-crit { background: linear-gradient(135deg,#3d0010,#5c0018); border:1px solid #ff2244; border-radius:12px; padding:1.5rem; text-align:center; }
# .result-title { font-size: 1.4rem; font-weight: 700; letter-spacing: 2px; margin: 0; }
# .result-sub   { font-size: 0.85rem; margin-top: 0.4rem; opacity: 0.8; }

# /* Séparateur */
# hr { border-color: #1e2d45; }
# </style>
# """, unsafe_allow_html=True)

# # ─── Constantes ───────────────────────────────────────────────
# CLASS_NAMES = [
#     "sain", "barre_1", "barre_1_par_pole",
#     "barre_2_opposees", "barre_4_adjacentes",
#     "exc_20", "exc_40", "exc_60", "mixte"
# ]
# CLASS_LABELS = {
#     "sain":              "Machine saine",
#     "barre_1":           "1 barre cassée",
#     "barre_1_par_pole":  "1 barre/pôle",
#     "barre_2_opposees":  "2 barres opposées",
#     "barre_4_adjacentes":"4 barres adjacentes",
#     "exc_20":            "Excentricité 20%",
#     "exc_40":            "Excentricité 40%",
#     "exc_60":            "Excentricité 60%",
#     "mixte":             "Défaut mixte",
# }
# SEVERITY = {
#     "sain": 0, "barre_1": 25, "barre_1_par_pole": 35,
#     "barre_2_opposees": 50, "barre_4_adjacentes": 65,
#     "exc_20": 20, "exc_40": 40, "exc_60": 65, "mixte": 85,
# }
# ACTION = {
#     "sain":              "✅ Aucune intervention requise.",
#     "barre_1":           "🔶 Surveiller — inspection sous 30 jours.",
#     "barre_1_par_pole":  "🔶 Maintenance planifiée sous 30 jours.",
#     "barre_2_opposees":  "⚠️ Maintenance préventive sous 7 jours.",
#     "barre_4_adjacentes":"🔴 Maintenance préventive sous 7 jours.",
#     "exc_20":            "🔶 Surveillance — vérifier alignement.",
#     "exc_40":            "⚠️ Réajustement de l'alignement requis.",
#     "exc_60":            "🔴 Risque contact rotor/stator — intervenir.",
#     "mixte":             "🚨 ARRÊT RECOMMANDÉ — intervention immédiate.",
# }

# # ─── Chargement modèle + pipeline (mis en cache) ──────────────
# @st.cache_resource
# def load_pipeline():
#     """
#     Charge le pipeline DiagMAS complet.
#     Utilise DiagnosticPipeline du projet pour garantir la cohérence
#     entre les features d'entraînement et celles d'inférence.
#     """
#     try:
#         from inference.predict import DiagnosticPipeline
#         pipe = DiagnosticPipeline.load()
#         return pipe, True
#     except Exception as e:
#         st.error(f"Erreur chargement pipeline: {e}")
#         return None, False

# @st.cache_resource
# def load_signal_generator():
#     from data_engine.signal_generator import SignalGenerator, FaultParams
#     from configs.settings import DATA_CFG
#     gen = SignalGenerator(DATA_CFG, seed=42)
#     return gen, FaultParams, DATA_CFG

# pipeline, model_ok = load_pipeline()

# if not model_ok or pipeline is None:
#     st.error("❌ Modèle non trouvé. Lancez d'abord : `python main.py train`")
#     st.stop()

# gen, FaultParams, DATA_CFG = load_signal_generator()


# # ─── Helpers ─────────────────────────────────────────────────
# def severity_color(sev: float) -> str:
#     if sev < 20:  return "#39ff14"
#     if sev < 50:  return "#ffbb00"
#     return "#ff2244"

# def result_css_class(sev: float) -> str:
#     if sev < 20:  return "result-ok"
#     if sev < 50:  return "result-warn"
#     return "result-crit"

# def result_icon(cls: str) -> str:
#     if cls == "sain": return "✅"
#     if "barre" in cls: return "⚙️"
#     if "exc" in cls:   return "🔄"
#     return "⚠️"

# def run_diagnosis(signal: np.ndarray) -> dict:
#     """Lance le diagnostic sur un signal numpy."""
#     return pipeline.diagnose(signal.astype(np.float32))

# def make_fault_params(cls_name: str, load: float = 1.0) -> object:
#     """Construit FaultParams pour une classe donnée."""
#     mapping = {
#         "sain":              FaultParams(load=load),
#         "barre_1":           FaultParams(n_broken=1, adjacent=True,  load=load),
#         "barre_1_par_pole":  FaultParams(n_broken=2, adjacent=False, load=load),
#         "barre_2_opposees":  FaultParams(n_broken=4, adjacent=False, load=load),
#         "barre_4_adjacentes":FaultParams(n_broken=4, adjacent=True,  load=load),
#         "exc_20":            FaultParams(eccentricity=0.20, load=load),
#         "exc_40":            FaultParams(eccentricity=0.40, load=load),
#         "exc_60":            FaultParams(eccentricity=0.60, load=load),
#         "mixte":             FaultParams(n_broken=4, adjacent=True, eccentricity=0.60, load=load),
#     }
#     return mapping[cls_name]


# # ─── Graphiques ───────────────────────────────────────────────
# # NOTE : xaxis / yaxis sont intentionnellement ABSENTS de PLOTLY_DARK
# # car chaque fonction passe ses propres options (title, range…).
# # Les mettre ici causerait "multiple values for keyword argument xaxis".
# PLOTLY_DARK = dict(
#     paper_bgcolor="#0a0d12", plot_bgcolor="#0f1520",
#     font=dict(color="#c8d8e8", family="monospace", size=11),
#     margin=dict(l=40, r=20, t=40, b=40),
# )
# # Grille commune appliquée via xaxis/yaxis dans chaque fonction
# _AXIS_STYLE = dict(gridcolor="#1e2d45", linecolor="#1e2d45", color="#5a7090")

# def plot_signal(signal: np.ndarray, fs: float = 10000.0,
#                 title: str = "Courant statorique — Phase A") -> go.Figure:
#     # Sous-échantillonnage + conversion float64 (requis par Plotly)
#     stride  = max(1, len(signal) // 3000)
#     sig_plt = signal[::stride].astype(float)
#     t_plt   = (np.arange(len(signal))[::stride] / fs).astype(float)
#     t_max   = float(t_plt[-1])
#     y_peak  = max(float(np.max(np.abs(sig_plt))) * 1.15, 0.5)

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=t_plt.tolist(),
#         y=sig_plt.tolist(),
#         mode="lines",
#         line=dict(color="#00d4ff", width=0.8),
#         name="i_a(t)"
#     ))
#     fig.update_layout(
#         **PLOTLY_DARK,
#         title=dict(text=title, font=dict(color="#00d4ff", size=12)),
#         xaxis=dict(**_AXIS_STYLE, title="Temps [s]",
#                    range=[0, t_max], autorange=False),
#         yaxis=dict(**_AXIS_STYLE, title="Courant [A]",
#                    range=[-y_peak, y_peak], autorange=False),
#         height=240,
#     )
#     return fig

# def plot_fft(signal: np.ndarray, fs: float = 10000.0,
#              g: float = 0.04, title: str = "Spectre FFT MCSA") -> go.Figure:
#     from features.extractor import FeatureExtractor
#     ext = FeatureExtractor(fs=fs)
#     freqs, amp, amp_db = ext._compute_fft(signal)
#     fn = 50.0
#     fr = fn * (1 - g) / 2

#     mask = freqs <= 120
#     # Conversion float64 obligatoire pour que Plotly affiche les données
#     freqs_plt  = freqs[mask].astype(float).tolist()
#     amp_db_plt = amp_db[mask].astype(float).tolist()
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=freqs_plt, y=amp_db_plt, mode="lines",
#         line=dict(color="#ffbb00", width=0.9), name="Spectre"
#     ))
#     # Annotations
#     annotations_data = [
#         (fn,              "#00d4ff", f"fs={fn}Hz"),
#         ((1-2*g)*fn,      "#ff6b35", f"f_b⁻={((1-2*g)*fn):.1f}Hz"),
#         ((1+2*g)*fn,      "#ff6b35", f"f_b⁺={((1+2*g)*fn):.1f}Hz"),
#         (fn - fr,         "#ff2244", f"f_e⁻={(fn-fr):.1f}Hz"),
#         (fn + fr,         "#ff2244", f"f_e⁺={(fn+fr):.1f}Hz"),
#     ]
#     for f_ann, color, label in annotations_data:
#         if 0 < f_ann <= 120:
#             fig.add_vline(x=f_ann, line=dict(color=color, width=0.8, dash="dash"))
#             fig.add_annotation(
#                 x=f_ann, y=-20, text=label, textangle=-90,
#                 font=dict(color=color, size=8), showarrow=False, yanchor="top"
#             )
#     fig.update_layout(
#         **PLOTLY_DARK,
#         title=dict(text=title, font=dict(color="#ffbb00", size=12)),
#         xaxis=dict(**_AXIS_STYLE, title="Fréquence [Hz]",
#                    range=[0, 120], autorange=False),
#         yaxis=dict(**_AXIS_STYLE, title="Amplitude [dB]",
#                    range=[-90, 10], autorange=False),
#         height=240,
#     )
#     return fig

# def plot_probas(all_probas: dict) -> go.Figure:
#     labels = [CLASS_LABELS[k] for k in CLASS_NAMES]
#     values = [all_probas.get(k, 0) * 100 for k in CLASS_NAMES]
#     colors = [severity_color(SEVERITY.get(k, 0)) for k in CLASS_NAMES]
#     fig = go.Figure(go.Bar(
#         x=values, y=labels, orientation="h",
#         marker=dict(color=colors, line=dict(color="#1e2d45", width=1)),
#         text=[f"{v:.1f}%" for v in values],
#         textposition="outside",
#     ))
#     fig.update_layout(
#         **PLOTLY_DARK,
#         title=dict(text="Probabilités par classe", font=dict(color="#c8d8e8", size=12)),
#         xaxis=dict(**_AXIS_STYLE, title="Probabilité (%)", range=[0, 115]),
#         yaxis=dict(**_AXIS_STYLE),
#         height=320,
#     )
#     return fig

# def plot_severity_gauge(severity: float) -> go.Figure:
#     color = severity_color(severity)
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=severity,
#         domain={"x": [0, 1], "y": [0, 1]},
#         title={"text": "Sévérité", "font": {"color": "#c8d8e8", "size": 13}},
#         number={"suffix": "%", "font": {"color": color, "size": 28}},
#         gauge={
#             "axis": {"range": [0, 100], "tickcolor": "#5a7090"},
#             "bar":  {"color": color},
#             "bgcolor": "#0f1520",
#             "bordercolor": "#1e2d45",
#             "steps": [
#                 {"range": [0, 20],  "color": "#0d1a0d"},
#                 {"range": [20, 50], "color": "#1a1a0d"},
#                 {"range": [50, 100],"color": "#1a0d0d"},
#             ],
#             "threshold": {
#                 "line": {"color": "#ff2244", "width": 2},
#                 "thickness": 0.75, "value": 75
#             },
#         }
#     ))
#     fig.update_layout(paper_bgcolor="#0a0d12", font=dict(color="#c8d8e8"),
#                       height=220, margin=dict(l=20,r=20,t=50,b=10))
#     return fig


# # ─────────────────────────────────────────────────────────────
# # HEADER
# # ─────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="main-header">
#   <h1>⚙️ DiagMAS-AI — DIAGNOSTIC MOTEUR ASYNCHRONE</h1>
#   <p>Machine 2.2 kW / 220 V / 50 Hz / 1440 tr/min &nbsp;|&nbsp;
#      Modèle IA : Random Forest + SVM + Gradient Boost &nbsp;|&nbsp;
#      Précision : 97.5% &nbsp;|&nbsp; 9 classes de défauts</p>
# </div>
# """, unsafe_allow_html=True)

# # ─────────────────────────────────────────────────────────────
# # SIDEBAR
# # ─────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## ⚙️ DiagMAS-AI")
#     st.markdown("---")
#     mode = st.radio(
#         "Mode",
#         ["📊 Simulation", "🎥 Démo toutes classes", "📁 Upload CSV", "📈 Comparaison"],
#         label_visibility="collapsed"
#     )
#     st.markdown("---")
#     st.markdown("**Moteur étudié**")
#     for k, v in [("Puissance","2.2 kW"), ("Tension","220 V (Y)"),
#                   ("Vitesse","1440 tr/min"), ("Fréquence","50 Hz"),
#                   ("Pôles","4 (2 paires)"), ("Barres rotor","48"),
#                   ("Encoches stat.","36"), ("Entrefer","0.33 mm")]:
#         st.markdown(f"<small style='color:#5a7090'>{k}</small>&nbsp;&nbsp;"
#                     f"<small style='color:#00d4ff'><b>{v}</b></small>", unsafe_allow_html=True)
#     st.markdown("---")
#     st.markdown(f"<small style='color:#5a7090'>Modèle chargé ✓<br>"
#                 f"{len(pipeline.model.feature_names)} features | 9 classes</small>",
#                 unsafe_allow_html=True)


# # ═════════════════════════════════════════════════════════════
# # MODE 1 — SIMULATION
# # ═════════════════════════════════════════════════════════════
# if mode == "📊 Simulation":
#     st.subheader("Simulation d'un défaut — signal physique réel")

#     col_ctrl, col_res = st.columns([1, 2])

#     with col_ctrl:
#         cls_choice = st.selectbox(
#             "Type de défaut",
#             CLASS_NAMES,
#             format_func=lambda x: CLASS_LABELS[x]
#         )
#         load_pct = st.slider("Charge mécanique (%)", 60, 100, 100)
#         seed_val = st.number_input("Seed (reproductibilité)", 0, 9999, 42)

#         st.markdown("---")
#         run_btn = st.button("▶ Lancer le diagnostic", use_container_width=True,
#                             type="primary")

#     if run_btn:
#         with st.spinner("Génération du signal et diagnostic..."):
#             t0 = time.perf_counter()
#             from data_engine.signal_generator import SignalGenerator as SG
#             g_loc = SG(DATA_CFG, seed=int(seed_val))
#             fp    = make_fault_params(cls_choice, load=load_pct / 100)
#             sig   = g_loc.generate(fp)
#             result = run_diagnosis(sig)
#             elapsed_ms = (time.perf_counter() - t0) * 1000

#         sev       = result["severity"]
#         css_cls   = result_css_class(sev)
#         icon      = result_icon(result["class_name"])
#         clabel    = CLASS_LABELS[result["class_name"]]
#         color     = severity_color(sev)

#         with col_res:
#             # Résultat principal
#             st.markdown(f"""
#             <div class="{css_cls}">
#               <div class="result-title" style="color:{color}">
#                 {icon} {clabel.upper()}
#               </div>
#               <div class="result-sub">
#                 Confiance : <b>{result['confidence']*100:.1f}%</b> &nbsp;|&nbsp;
#                 Sévérité : <b>{sev:.0f}%</b> — {result['severity_label']}
#               </div>
#               <div class="result-sub" style="margin-top:0.5rem">
#                 {ACTION.get(result['class_name'], '')}
#               </div>
#               <div style="font-size:0.72rem;color:#5a7090;margin-top:0.4rem">
#                 Latence : {elapsed_ms:.1f} ms
#               </div>
#             </div>""", unsafe_allow_html=True)

#         # Graphiques
#         c1, c2 = st.columns(2)
#         with c1:
#             st.plotly_chart(plot_signal(sig), use_container_width=True)
#         with c2:
#             g_est = result.get("estimated_slip", 0.04)
#             st.plotly_chart(plot_fft(sig, g=result.get("estimated_slip", 0.04)),
#                             use_container_width=True)

#         c3, c4 = st.columns([2, 1])
#         with c3:
#             st.plotly_chart(plot_probas(result["all_probas"]), use_container_width=True)
#         with c4:
#             st.plotly_chart(plot_severity_gauge(sev), use_container_width=True)

#         # Tableau des features clés
#         with st.expander("📋 Features MCSA extraites"):
#             feat_names = pipeline.model.feature_names
#             from features.extractor import FeatureExtractor
#             ext = FeatureExtractor(fs=DATA_CFG.fs)
#             x_vec = ext.extract(sig)
#             feat_df = pd.DataFrame({
#                 "Feature": feat_names,
#                 "Valeur":  [f"{v:.4f}" for v in x_vec]
#             })
#             # Top features importantes
#             barre_feats = feat_df[feat_df["Feature"].str.contains("barre")]
#             exc_feats   = feat_df[feat_df["Feature"].str.contains("exc")]
#             time_feats  = feat_df[feat_df["Feature"].str.contains("rms|kurtosis|crest|skew")]
#             st.markdown("**Features barres cassées**")
#             st.dataframe(barre_feats, hide_index=True, use_container_width=True)
#             st.markdown("**Features excentricité**")
#             st.dataframe(exc_feats, hide_index=True, use_container_width=True)
#             st.markdown("**Features temporelles**")
#             st.dataframe(time_feats, hide_index=True, use_container_width=True)

#     else:
#         with col_res:
#             st.info("Sélectionnez un type de défaut et cliquez sur **▶ Lancer le diagnostic**")


# # ═════════════════════════════════════════════════════════════
# # MODE 2 — DÉMO TOUTES CLASSES
# # ═════════════════════════════════════════════════════════════
# elif mode == "🎥 Démo toutes classes":
#     st.subheader("Démonstration — Les 9 classes de défauts")
#     st.markdown("Chaque signal est généré par le **vrai modèle physique** du projet "
#                 "(générateur d-q calibré sur les paramètres du mémoire).")

#     if st.button("▶ Lancer la démo complète", type="primary", use_container_width=True):
#         from data_engine.signal_generator import SignalGenerator as SG
#         g_demo = SG(DATA_CFG, seed=100)

#         progress_bar = st.progress(0, text="Initialisation...")
#         results_table = []
#         all_correct = 0

#         for i, cls_name in enumerate(CLASS_NAMES):
#             progress_bar.progress((i + 1) / len(CLASS_NAMES),
#                                    text=f"Analyse : {CLASS_LABELS[cls_name]}...")
#             fp  = make_fault_params(cls_name, load=1.0)
#             sig = g_demo.generate(fp)
#             r   = run_diagnosis(sig)

#             correct = r["class_name"] == cls_name
#             if correct: all_correct += 1
#             color = severity_color(r["severity"])
#             icon  = "✓" if correct else "✗"

#             results_table.append({
#                 "":            icon,
#                 "Classe réelle": CLASS_LABELS[cls_name],
#                 "Prédiction":  CLASS_LABELS[r["class_name"]],
#                 "Confiance":   f"{r['confidence']*100:.0f}%",
#                 "Sévérité":    f"{r['severity']:.0f}%",
#                 "Verdict":     r["severity_label"],
#                 "Action":      ACTION.get(r["class_name"], "")[:55],
#             })

#         progress_bar.empty()
#         df_res = pd.DataFrame(results_table)

#         acc = all_correct / len(CLASS_NAMES) * 100
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Classes testées", len(CLASS_NAMES))
#         c2.metric("Correctes", f"{all_correct}/{len(CLASS_NAMES)}")
#         c3.metric("Précision démo", f"{acc:.0f}%")

#         # Colorier le tableau
#         def style_row(row):
#             if row[""] == "✓":
#                 return ["color: #39ff14"] * len(row)
#             return ["color: #ff2244"] * len(row)

#         st.dataframe(
#             df_res.style.apply(style_row, axis=1),
#             use_container_width=True, hide_index=True
#         )

#         if acc == 100:
#             st.success("🎯 100% correct — Toutes les classes correctement identifiées !")
#             st.balloons()
#         elif acc >= 80:
#             st.success(f"✅ {acc:.0f}% correct")
#         else:
#             st.warning(f"⚠️ {acc:.0f}% correct — vérifier les paramètres du modèle")

#     else:
#         st.info("Cliquez sur **▶ Lancer la démo complète** pour tester les 9 classes.")


# # ═════════════════════════════════════════════════════════════
# # MODE 3 — UPLOAD CSV
# # ═════════════════════════════════════════════════════════════
# elif mode == "📁 Upload CSV":
#     st.subheader("Diagnostic depuis un fichier CSV")
#     st.markdown("""
#     **Format attendu :** un fichier CSV avec une colonne de courant statorique.
#     - Fréquence d'échantillonnage : **10 000 Hz** (configurable)
#     - Durée minimale : **0.5 seconde** (5 000 points)
#     - La colonne doit s'appeler `courant`, `signal`, `current` ou être la première colonne.
#     """)

#     col_u, col_cfg = st.columns([2, 1])
#     with col_cfg:
#         fs_input = st.number_input("Fréquence d'échantillonnage [Hz]", 1000, 50000, 10000)
#         col_name = st.text_input("Nom de la colonne signal (vide = auto)", "")

#     with col_u:
#         uploaded = st.file_uploader("Déposer un fichier CSV", type=["csv", "txt"])

#     if uploaded:
#         try:
#             df = pd.read_csv(uploaded)
#             st.success(f"Fichier chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

#             # Identifier la colonne signal
#             if col_name and col_name in df.columns:
#                 sig_col = col_name
#             else:
#                 candidates = [c for c in df.columns
#                               if any(k in c.lower() for k in
#                                      ["courant", "signal", "current", "i_a", "phase_a"])]
#                 sig_col = candidates[0] if candidates else df.columns[0]
#                 st.info(f"Colonne utilisée : **{sig_col}**")

#             signal_raw = df[sig_col].dropna().values.astype(np.float32)
#             min_len    = int(fs_input * 0.5)

#             if len(signal_raw) < min_len:
#                 st.error(f"Signal trop court : {len(signal_raw)} pts "
#                           f"(minimum {min_len} pour {fs_input} Hz)")
#                 st.stop()

#             # Aperçu
#             from features.extractor import FeatureExtractor
#             ext_up = FeatureExtractor(fs=float(fs_input))
#             st.plotly_chart(
#                 plot_signal(signal_raw, fs=fs_input, title=f"Signal : {sig_col}"),
#                 use_container_width=True
#             )

#             if st.button("🔍 Lancer le diagnostic", type="primary", use_container_width=True):
#                 with st.spinner("Extraction des features et diagnostic..."):
#                     # On crée un pipeline temporaire avec la bonne fs
#                     from inference.predict import DiagnosticPipeline
#                     tmp_pipe = DiagnosticPipeline(model=pipeline.model, fs=float(fs_input))
#                     r = tmp_pipe.diagnose(signal_raw)

#                 sev   = r["severity"]
#                 color = severity_color(sev)
#                 clss  = result_css_class(sev)

#                 st.markdown(f"""
#                 <div class="{clss}" style="margin:1rem 0">
#                   <div class="result-title" style="color:{color}">
#                     {result_icon(r['class_name'])} {CLASS_LABELS[r['class_name']].upper()}
#                   </div>
#                   <div class="result-sub">
#                     Confiance : <b>{r['confidence']*100:.1f}%</b> &nbsp;|&nbsp;
#                     Sévérité : <b>{sev:.0f}%</b> — {r['severity_label']}
#                   </div>
#                   <div class="result-sub">{ACTION.get(r['class_name'],'')}</div>
#                 </div>""", unsafe_allow_html=True)

#                 c1, c2 = st.columns(2)
#                 with c1:
#                     st.plotly_chart(
#                         plot_fft(signal_raw, fs=fs_input, title="Spectre FFT MCSA"),
#                         use_container_width=True
#                     )
#                 with c2:
#                     st.plotly_chart(plot_probas(r["all_probas"]), use_container_width=True)

#         except Exception as e:
#             st.error(f"Erreur lecture fichier : {e}")


# # ═════════════════════════════════════════════════════════════
# # MODE 4 — COMPARAISON
# # ═════════════════════════════════════════════════════════════
# elif mode == "📈 Comparaison":
#     st.subheader("Comparaison côte à côte de deux défauts")

#     c1, c2 = st.columns(2)
#     with c1:
#         cls_a = st.selectbox("Défaut A", CLASS_NAMES,
#                               format_func=lambda x: CLASS_LABELS[x], key="ca")
#     with c2:
#         cls_b = st.selectbox("Défaut B", CLASS_NAMES, index=4,
#                               format_func=lambda x: CLASS_LABELS[x], key="cb")

#     if st.button("⚖️ Comparer", type="primary", use_container_width=True):
#         from data_engine.signal_generator import SignalGenerator as SG
#         g_cmp = SG(DATA_CFG, seed=77)

#         with st.spinner("Génération et analyse..."):
#             sig_a = g_cmp.generate(make_fault_params(cls_a))
#             sig_b = g_cmp.generate(make_fault_params(cls_b))
#             r_a   = run_diagnosis(sig_a)
#             r_b   = run_diagnosis(sig_b)

#         # KPIs côte à côte
#         ca, cb = st.columns(2)
#         for col, r, cls, sig in [(ca, r_a, cls_a, sig_a), (cb, r_b, cls_b, sig_b)]:
#             sev   = r["severity"]
#             color = severity_color(sev)
#             with col:
#                 st.markdown(f"""
#                 <div class="{result_css_class(sev)}" style="margin-bottom:1rem">
#                   <div class="result-title" style="color:{color}">
#                     {result_icon(r['class_name'])} {CLASS_LABELS[r['class_name']].upper()}
#                   </div>
#                   <div class="result-sub">
#                     Confiance : {r['confidence']*100:.0f}% &nbsp;|&nbsp;
#                     Sévérité : {sev:.0f}%
#                   </div>
#                 </div>""", unsafe_allow_html=True)
#                 st.plotly_chart(
#                     plot_signal(sig, title=CLASS_LABELS[cls]),
#                     use_container_width=True
#                 )
#                 st.plotly_chart(
#                     plot_fft(sig, title="Spectre FFT"),
#                     use_container_width=True
#                 )

#         # Comparaison probabilités
#         st.markdown("### Comparaison des probabilités")
#         labels = [CLASS_LABELS[k] for k in CLASS_NAMES]
#         vals_a = [r_a["all_probas"].get(k, 0)*100 for k in CLASS_NAMES]
#         vals_b = [r_b["all_probas"].get(k, 0)*100 for k in CLASS_NAMES]

#         fig_cmp = go.Figure()
#         fig_cmp.add_trace(go.Bar(name=CLASS_LABELS[cls_a], x=labels,
#                                   y=[float(v) for v in vals_a],
#                                   marker_color="#00d4ff"))
#         fig_cmp.add_trace(go.Bar(name=CLASS_LABELS[cls_b], x=labels,
#                                   y=[float(v) for v in vals_b],
#                                   marker_color="#ff6b35"))
#         fig_cmp.update_layout(
#             **PLOTLY_DARK, barmode="group",
#             xaxis=dict(**_AXIS_STYLE, tickangle=-35),
#             yaxis=dict(**_AXIS_STYLE, title="Probabilité (%)"),
#             height=350,
#         )
#         st.plotly_chart(fig_cmp, use_container_width=True)


# # ─── Footer ───────────────────────────────────────────────────
# st.markdown("---")
# st.markdown(
#     "<div style='text-align:center;color:#5a7090;font-size:0.75rem;letter-spacing:1px'>"
#     "DiagMAS-AI &nbsp;·&nbsp; Université Mohamed Boudiaf M'Sila &nbsp;·&nbsp; "
#     "Diagnostic des défauts dans le moteur asynchrone &nbsp;·&nbsp; 2024-2025"
#     "</div>",
#     unsafe_allow_html=True
# )

"""
app_streamlit.py — DiagMAS-AI avec choix entre deux modèles
- Modèle standard (synthétique)
- Modèle fine-tuné (synthétique + CWRU)
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import time

# ─── Config page ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiagMAS-AI — Diagnostic Moteur Asynchrone",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0d12; color: #c8d8e8; }
section[data-testid="stSidebar"] { background-color: #0f1520; border-right: 1px solid #1e2d45; }

.main-header {
    background: linear-gradient(135deg, #001833 0%, #003d5c 100%);
    padding: 1.5rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #00d4ff44;
    box-shadow: 0 4px 32px rgba(0,212,255,0.1);
}
.main-header h1 { color: #00d4ff; margin: 0; font-size: 1.8rem; letter-spacing: 2px; }
.main-header p  { color: #5a7090; margin: 0.4rem 0 0 0; font-size: 0.9rem; letter-spacing: 1px; }

.kpi-card {
    background: #0f1520; border: 1px solid #1e2d45;
    border-radius: 10px; padding: 1rem 1.2rem;
    text-align: center;
}
.kpi-title { font-size: 0.75rem; color: #5a7090; letter-spacing: 2px; text-transform: uppercase; }
.kpi-value { font-size: 2rem; font-weight: 700; font-family: monospace; }
.kpi-unit  { font-size: 0.75rem; color: #5a7090; }

.result-ok   { background: linear-gradient(135deg,#003d1a,#005c28); border:1px solid #39ff14; border-radius:12px; padding:1.5rem; text-align:center; }
.result-warn { background: linear-gradient(135deg,#3d2e00,#5c4400); border:1px solid #ffbb00; border-radius:12px; padding:1.5rem; text-align:center; }
.result-crit { background: linear-gradient(135deg,#3d0010,#5c0018); border:1px solid #ff2244; border-radius:12px; padding:1.5rem; text-align:center; }
.result-title { font-size: 1.4rem; font-weight: 700; letter-spacing: 2px; margin: 0; }
.result-sub   { font-size: 0.85rem; margin-top: 0.4rem; opacity: 0.8; }

hr { border-color: #1e2d45; }
</style>
""", unsafe_allow_html=True)

# ─── Constantes ───────────────────────────────────────────────
CLASS_NAMES = [
    "sain", "barre_1", "barre_1_par_pole",
    "barre_2_opposees", "barre_4_adjacentes",
    "exc_20", "exc_40", "exc_60", "mixte"
]
CLASS_LABELS = {
    "sain":              "Machine saine",
    "barre_1":           "1 barre cassée",
    "barre_1_par_pole":  "1 barre/pôle",
    "barre_2_opposees":  "2 barres opposées",
    "barre_4_adjacentes":"4 barres adjacentes",
    "exc_20":            "Excentricité 20%",
    "exc_40":            "Excentricité 40%",
    "exc_60":            "Excentricité 60%",
    "mixte":             "Défaut mixte",
}
SEVERITY = {
    "sain": 0, "barre_1": 25, "barre_1_par_pole": 35,
    "barre_2_opposees": 50, "barre_4_adjacentes": 65,
    "exc_20": 20, "exc_40": 40, "exc_60": 65, "mixte": 85,
}
ACTION = {
    "sain":              "✅ Aucune intervention requise.",
    "barre_1":           "🔶 Surveiller — inspection sous 30 jours.",
    "barre_1_par_pole":  "🔶 Maintenance planifiée sous 30 jours.",
    "barre_2_opposees":  "⚠️ Maintenance préventive sous 7 jours.",
    "barre_4_adjacentes":"🔴 Maintenance préventive sous 7 jours.",
    "exc_20":            "🔶 Surveillance — vérifier alignement.",
    "exc_40":            "⚠️ Réajustement de l'alignement requis.",
    "exc_60":            "🔴 Risque contact rotor/stator — intervenir.",
    "mixte":             "🚨 ARRÊT RECOMMANDÉ — intervention immédiate.",
}

# ─── Chargement des DEUX modèles ──────────────────────────────
# def load_both_models():
#     """
#     Charge les deux modèles :
#     - standard : entraîné sur données synthétiques
#     - finetuned : entraîné sur synthétique + CWRU
#     """
#     models = {}
    
#     # Modèle standard
#     try:
#         from inference.predict import DiagnosticPipeline
#         pipe_std = DiagnosticPipeline.load("outputs/models/diagmas_model.pkl")
#         models["standard"] = pipe_std
#     except Exception as e:
#         st.warning(f"Modèle standard non chargé: {e}")
#         models["standard"] = None
    
#     # Modèle fine-tuné
#     try:
#         pipe_ft = DiagnosticPipeline.load("outputs/models/diagmas_model_finetuned.pkl")
#         models["finetuned"] = pipe_ft
#     except Exception as e:
#         st.warning(f"Modèle fine-tuné non chargé: {e}")
#         models["finetuned"] = None
    
#     return models
# ─── Chargement des DEUX modèles (version simplifiée) ─────────
@st.cache_resource
def load_both_models():
    """
    Charge les deux modèles sans dépendre de DiagnosticPipeline
    """
    from features.extractor import FeatureExtractor
    
    class SimplePipelineWrapper:
        def __init__(self, model, class_names=None):
            self.model = model
            self.class_names = class_names or CLASS_NAMES
            self.feature_names = [f"feat_{i}" for i in range(42)]
        
        def diagnose(self, signal):
            ext = FeatureExtractor(fs=10000)
            x = ext.extract(signal).reshape(1, -1)
            pred_idx = int(self.model.predict(x)[0])
            probas = self.model.predict_proba(x)[0]
            confidence = float(probas[pred_idx])
            
            severity_map = {0:0, 1:25, 2:35, 3:50, 4:65, 5:20, 6:40, 7:65, 8:85}
            severity = severity_map.get(pred_idx, 0) * confidence
            
            return {
                'class_name': self.class_names[pred_idx],
                'confidence': confidence,
                'severity': severity,
                'severity_label': 'Normal' if severity < 20 else 'Attention' if severity < 50 else 'CRITIQUE',
                'all_probas': {self.class_names[i]: float(probas[i]) for i in range(len(self.class_names))},
                'estimated_slip': 0.04
            }
    
    models = {}
    
    # Modèle standard
    try:
        obj = joblib.load("outputs/models/diagmas_model.pkl")
        if isinstance(obj, dict):
            pipeline = obj.get('pipeline')
        else:
            pipeline = obj
        
        if pipeline is not None:
            models["standard"] = SimplePipelineWrapper(pipeline)
            print("✅ Modèle standard chargé")
        else:
            models["standard"] = None
    except Exception as e:
        st.warning(f"Modèle standard non chargé: {e}")
        models["standard"] = None
    
    # Modèle fine-tuné
    try:
        obj = joblib.load("outputs/models/diagmas_model_finetuned2.pkl")
        if isinstance(obj, dict):
            pipeline = obj.get('pipeline')
        else:
            pipeline = obj
        
        if pipeline is not None:
            models["finetuned"] = SimplePipelineWrapper(pipeline)
            print("✅ Modèle fine-tuné chargé")
        else:
            models["finetuned"] = None
    except Exception as e:
        st.warning(f"Modèle fine-tuné non chargé: {e}")
        models["finetuned"] = None
    
    return models

@st.cache_resource
def load_signal_generator():
    from data_engine.signal_generator import SignalGenerator, FaultParams
    from configs.settings import DATA_CFG
    gen = SignalGenerator(DATA_CFG, seed=42)
    return gen, FaultParams, DATA_CFG

models = load_both_models()
gen, FaultParams, DATA_CFG = load_signal_generator()

# Vérifier qu'au moins un modèle est chargé
if models["standard"] is None and models["finetuned"] is None:
    st.error("❌ Aucun modèle trouvé. Lancez d'abord : `python main.py train`")
    st.stop()

# ─── Helpers ─────────────────────────────────────────────────
def severity_color(sev: float) -> str:
    if sev < 20:  return "#39ff14"
    if sev < 50:  return "#ffbb00"
    return "#ff2244"

def result_css_class(sev: float) -> str:
    if sev < 20:  return "result-ok"
    if sev < 50:  return "result-warn"
    return "result-crit"

def result_icon(cls: str) -> str:
    if cls == "sain": return "✅"
    if "barre" in cls: return "⚙️"
    if "exc" in cls:   return "🔄"
    return "⚠️"

def run_diagnosis_with_model(pipeline, signal: np.ndarray) -> dict:
    """Lance le diagnostic sur un signal avec un modèle donné."""
    return pipeline.diagnose(signal.astype(np.float32))

def make_fault_params(cls_name: str, load: float = 1.0) -> object:
    mapping = {
        "sain":              FaultParams(load=load),
        "barre_1":           FaultParams(n_broken=1, adjacent=True,  load=load),
        "barre_1_par_pole":  FaultParams(n_broken=2, adjacent=False, load=load),
        "barre_2_opposees":  FaultParams(n_broken=4, adjacent=False, load=load),
        "barre_4_adjacentes":FaultParams(n_broken=4, adjacent=True,  load=load),
        "exc_20":            FaultParams(eccentricity=0.20, load=load),
        "exc_40":            FaultParams(eccentricity=0.40, load=load),
        "exc_60":            FaultParams(eccentricity=0.60, load=load),
        "mixte":             FaultParams(n_broken=4, adjacent=True, eccentricity=0.60, load=load),
    }
    return mapping[cls_name]

# ─── Graphiques ───────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor="#0a0d12", plot_bgcolor="#0f1520",
    font=dict(color="#c8d8e8", family="monospace", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)
_AXIS_STYLE = dict(gridcolor="#1e2d45", linecolor="#1e2d45", color="#5a7090")

def plot_signal(signal: np.ndarray, fs: float = 10000.0,
                title: str = "Courant statorique — Phase A") -> go.Figure:
    stride  = max(1, len(signal) // 3000)
    sig_plt = signal[::stride].astype(float)
    t_plt   = (np.arange(len(signal))[::stride] / fs).astype(float)
    t_max   = float(t_plt[-1])
    y_peak  = max(float(np.max(np.abs(sig_plt))) * 1.15, 0.5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_plt.tolist(),
        y=sig_plt.tolist(),
        mode="lines",
        line=dict(color="#00d4ff", width=0.8),
        name="i_a(t)"
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text=title, font=dict(color="#00d4ff", size=12)),
        xaxis=dict(**_AXIS_STYLE, title="Temps [s]",
                   range=[0, t_max], autorange=False),
        yaxis=dict(**_AXIS_STYLE, title="Courant [A]",
                   range=[-y_peak, y_peak], autorange=False),
        height=240,
    )
    return fig

def plot_fft(signal: np.ndarray, fs: float = 10000.0,
             g: float = 0.04, title: str = "Spectre FFT MCSA") -> go.Figure:
    from features.extractor import FeatureExtractor
    ext = FeatureExtractor(fs=fs)
    freqs, amp, amp_db = ext._compute_fft(signal)
    fn = 50.0
    fr = fn * (1 - g) / 2

    mask = freqs <= 120
    freqs_plt  = freqs[mask].astype(float).tolist()
    amp_db_plt = amp_db[mask].astype(float).tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs_plt, y=amp_db_plt, mode="lines",
        line=dict(color="#ffbb00", width=0.9), name="Spectre"
    ))
    annotations_data = [
        (fn,              "#00d4ff", f"fs={fn}Hz"),
        ((1-2*g)*fn,      "#ff6b35", f"f_b⁻={((1-2*g)*fn):.1f}Hz"),
        ((1+2*g)*fn,      "#ff6b35", f"f_b⁺={((1+2*g)*fn):.1f}Hz"),
        (fn - fr,         "#ff2244", f"f_e⁻={(fn-fr):.1f}Hz"),
        (fn + fr,         "#ff2244", f"f_e⁺={(fn+fr):.1f}Hz"),
    ]
    for f_ann, color, label in annotations_data:
        if 0 < f_ann <= 120:
            fig.add_vline(x=f_ann, line=dict(color=color, width=0.8, dash="dash"))
            fig.add_annotation(
                x=f_ann, y=-20, text=label, textangle=-90,
                font=dict(color=color, size=8), showarrow=False, yanchor="top"
            )
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text=title, font=dict(color="#ffbb00", size=12)),
        xaxis=dict(**_AXIS_STYLE, title="Fréquence [Hz]",
                   range=[0, 120], autorange=False),
        yaxis=dict(**_AXIS_STYLE, title="Amplitude [dB]",
                   range=[-90, 10], autorange=False),
        height=240,
    )
    return fig

def plot_probas(all_probas: dict) -> go.Figure:
    labels = [CLASS_LABELS[k] for k in CLASS_NAMES]
    values = [all_probas.get(k, 0) * 100 for k in CLASS_NAMES]
    colors = [severity_color(SEVERITY.get(k, 0)) for k in CLASS_NAMES]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="#1e2d45", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text="Probabilités par classe", font=dict(color="#c8d8e8", size=12)),
        xaxis=dict(**_AXIS_STYLE, title="Probabilité (%)", range=[0, 115]),
        yaxis=dict(**_AXIS_STYLE),
        height=320,
    )
    return fig

def plot_severity_gauge(severity: float) -> go.Figure:
    color = severity_color(severity)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=severity,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Sévérité", "font": {"color": "#c8d8e8", "size": 13}},
        number={"suffix": "%", "font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#5a7090"},
            "bar":  {"color": color},
            "bgcolor": "#0f1520",
            "bordercolor": "#1e2d45",
            "steps": [
                {"range": [0, 20],  "color": "#0d1a0d"},
                {"range": [20, 50], "color": "#1a1a0d"},
                {"range": [50, 100],"color": "#1a0d0d"},
            ],
            "threshold": {
                "line": {"color": "#ff2244", "width": 2},
                "thickness": 0.75, "value": 75
            },
        }
    ))
    fig.update_layout(paper_bgcolor="#0a0d12", font=dict(color="#c8d8e8"),
                      height=220, margin=dict(l=20,r=20,t=50,b=10))
    return fig


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚙️ DiagMAS-AI — DIAGNOSTIC MOTEUR ASYNCHRONE</h1>
  <p>Machine 2.2 kW / 220 V / 50 Hz / 1440 tr/min &nbsp;|&nbsp;
     Modèle IA : Random Forest + SVM + Gradient Boost &nbsp;|&nbsp;
     Deux modèles disponibles : Standard (synthétique) / Fine-tuné (CWRU)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ DiagMAS-AI")
    st.markdown("---")
    
    # Sélection du modèle
    st.markdown("### 🤖 Choix du modèle")
    model_option = st.radio(
        "Modèle à utiliser",
        ["standard", "finetuned"],
        format_func=lambda x: "📊 Standard (synthétique)" if x == "standard" else "🎯 Fine-tuné (CWRU + synthétique)"
    )
    
    # Informations sur le modèle sélectionné
    if model_option == "standard":
        st.info("✅ Modèle standard – 97.6% sur données synthétiques")
    else:
        st.success("🎯 Modèle fine-tuné – 86% sur CWRU, 100% cohérence binaire")
    
    st.markdown("---")
    
    mode = st.radio(
        "Mode",
        ["📊 Simulation", "🎥 Démo toutes classes", "📁 Upload CSV", "📈 Comparaison"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**Moteur étudié**")
    for k, v in [("Puissance","2.2 kW"), ("Tension","220 V (Y)"),
                  ("Vitesse","1440 tr/min"), ("Fréquence","50 Hz"),
                  ("Pôles","4 (2 paires)"), ("Barres rotor","48"),
                  ("Encoches stat.","36"), ("Entrefer","0.33 mm")]:
        st.markdown(f"<small style='color:#5a7090'>{k}</small>&nbsp;&nbsp;"
                    f"<small style='color:#00d4ff'><b>{v}</b></small>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Afficher quel modèle est actif
    if models[model_option] is not None:
        feat_count = 42
        st.markdown(f"<small style='color:#00d4ff'>✓ Modèle actif : {model_option}<br>{feat_count} features | 9 classes</small>",
                    unsafe_allow_html=True)
    else:
        st.error(f"❌ Modèle {model_option} indisponible")


# Sélectionner le pipeline actif
active_pipeline = models[model_option]

if active_pipeline is None:
    st.error(f"❌ Le modèle '{model_option}' n'est pas disponible. Vérifie les fichiers dans outputs/models/")
    st.stop()


# ═════════════════════════════════════════════════════════════
# MODE 1 — SIMULATION
# ═════════════════════════════════════════════════════════════
if mode == "📊 Simulation":
    st.subheader("Simulation d'un défaut — signal physique réel")
    
    # Info sur le modèle utilisé
    st.info(f"🤖 Modèle utilisé : **{model_option.upper()}**")

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        cls_choice = st.selectbox(
            "Type de défaut",
            CLASS_NAMES,
            format_func=lambda x: CLASS_LABELS[x]
        )
        load_pct = st.slider("Charge mécanique (%)", 60, 100, 100)
        seed_val = st.number_input("Seed (reproductibilité)", 0, 9999, 42)

        st.markdown("---")
        run_btn = st.button("▶ Lancer le diagnostic", use_container_width=True,
                            type="primary")

    if run_btn:
        with st.spinner("Génération du signal et diagnostic..."):
            t0 = time.perf_counter()
            from data_engine.signal_generator import SignalGenerator as SG
            g_loc = SG(DATA_CFG, seed=int(seed_val))
            fp    = make_fault_params(cls_choice, load=load_pct / 100)
            sig   = g_loc.generate(fp)
            result = run_diagnosis_with_model(active_pipeline, sig)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        sev       = result["severity"]
        css_cls   = result_css_class(sev)
        icon      = result_icon(result["class_name"])
        clabel    = CLASS_LABELS[result["class_name"]]
        color     = severity_color(sev)

        with col_res:
            st.markdown(f"""
            <div class="{css_cls}">
              <div class="result-title" style="color:{color}">
                {icon} {clabel.upper()}
              </div>
              <div class="result-sub">
                Confiance : <b>{result['confidence']*100:.1f}%</b> &nbsp;|&nbsp;
                Sévérité : <b>{sev:.0f}%</b> — {result['severity_label']}
              </div>
              <div class="result-sub" style="margin-top:0.5rem">
                {ACTION.get(result['class_name'], '')}
              </div>
              <div style="font-size:0.72rem;color:#5a7090;margin-top:0.4rem">
                Latence : {elapsed_ms:.1f} ms
              </div>
            </div>""", unsafe_allow_html=True)

        # Graphiques
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_signal(sig), use_container_width=True)
        with c2:
            st.plotly_chart(plot_fft(sig, g=result.get("estimated_slip", 0.04)),
                            use_container_width=True)

        c3, c4 = st.columns([2, 1])
        with c3:
            st.plotly_chart(plot_probas(result["all_probas"]), use_container_width=True)
        with c4:
            st.plotly_chart(plot_severity_gauge(sev), use_container_width=True)


    else:
        with col_res:
            st.info("Sélectionnez un type de défaut et cliquez sur **▶ Lancer le diagnostic**")


# ═════════════════════════════════════════════════════════════
# MODE 2 — DÉMO TOUTES CLASSES
# ═════════════════════════════════════════════════════════════
elif mode == "🎥 Démo toutes classes":
    st.subheader("Démonstration — Les 9 classes de défauts")
    st.info(f"🤖 Modèle utilisé : **{model_option.upper()}**")
    st.markdown("Chaque signal est généré par le **vrai modèle physique** du projet.")

    if st.button("▶ Lancer la démo complète", type="primary", use_container_width=True):
        from data_engine.signal_generator import SignalGenerator as SG
        g_demo = SG(DATA_CFG, seed=100)

        progress_bar = st.progress(0, text="Initialisation...")
        results_table = []
        all_correct = 0

        for i, cls_name in enumerate(CLASS_NAMES):
            progress_bar.progress((i + 1) / len(CLASS_NAMES),
                                   text=f"Analyse : {CLASS_LABELS[cls_name]}...")
            fp  = make_fault_params(cls_name, load=1.0)
            sig = g_demo.generate(fp)
            r   = run_diagnosis_with_model(active_pipeline, sig)

            correct = r["class_name"] == cls_name
            if correct: all_correct += 1

            results_table.append({
                "Classe réelle": CLASS_LABELS[cls_name],
                "Prédiction":    CLASS_LABELS[r["class_name"]],
                "Confiance":     f"{r['confidence']*100:.0f}%",
                "Sévérité":      f"{r['severity']:.0f}%",
                "Verdict":       r["severity_label"],
            })

        progress_bar.empty()
        df_res = pd.DataFrame(results_table)

        acc = all_correct / len(CLASS_NAMES) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Classes testées", len(CLASS_NAMES))
        c2.metric("Correctes", f"{all_correct}/{len(CLASS_NAMES)}")
        c3.metric("Précision démo", f"{acc:.0f}%")

        st.dataframe(df_res, use_container_width=True, hide_index=True)

        if acc == 100:
            st.success("🎯 100% correct — Toutes les classes correctement identifiées !")
            st.balloons()
        elif acc >= 80:
            st.success(f"✅ {acc:.0f}% correct")
        else:
            st.warning(f"⚠️ {acc:.0f}% correct — vérifier les paramètres du modèle")

    else:
        st.info("Cliquez sur **▶ Lancer la démo complète** pour tester les 9 classes.")


# ═════════════════════════════════════════════════════════════
# MODE 3 — UPLOAD CSV
# ═════════════════════════════════════════════════════════════
elif mode == "📁 Upload CSV":
    st.subheader("Diagnostic depuis un fichier CSV")
    st.info(f"🤖 Modèle utilisé : **{model_option.upper()}**")
    st.markdown("""
    **Format attendu :** un fichier CSV avec une colonne de courant statorique.
    - Fréquence d'échantillonnage : **10 000 Hz** (configurable)
    - Durée minimale : **0.5 seconde** (5 000 points)
    - La colonne doit s'appeler `courant`, `signal`, `current` ou être la première colonne.
    """)

    col_u, col_cfg = st.columns([2, 1])
    with col_cfg:
        fs_input = st.number_input("Fréquence d'échantillonnage [Hz]", 1000, 50000, 10000)
        col_name = st.text_input("Nom de la colonne signal (vide = auto)", "")

    with col_u:
        uploaded = st.file_uploader("Déposer un fichier CSV", type=["csv", "txt"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Fichier chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

            if col_name and col_name in df.columns:
                sig_col = col_name
            else:
                candidates = [c for c in df.columns
                              if any(k in c.lower() for k in
                                     ["courant", "signal", "current", "i_a", "phase_a"])]
                sig_col = candidates[0] if candidates else df.columns[0]
                st.info(f"Colonne utilisée : **{sig_col}**")

            signal_raw = df[sig_col].dropna().values.astype(np.float32)
            min_len    = int(fs_input * 0.5)

            if len(signal_raw) < min_len:
                st.error(f"Signal trop court : {len(signal_raw)} pts "
                          f"(minimum {min_len} pour {fs_input} Hz)")
                st.stop()

            st.plotly_chart(
                plot_signal(signal_raw, fs=fs_input, title=f"Signal : {sig_col}"),
                use_container_width=True
            )

            if st.button("🔍 Lancer le diagnostic", type="primary", use_container_width=True):
                with st.spinner("Extraction des features et diagnostic..."):
                    from inference.predict import DiagnosticPipeline
                    tmp_pipe = DiagnosticPipeline(model=active_pipeline.model, fs=float(fs_input))
                    r = tmp_pipe.diagnose(signal_raw)

                sev   = r["severity"]
                color = severity_color(sev)
                clss  = result_css_class(sev)

                st.markdown(f"""
                <div class="{clss}" style="margin:1rem 0">
                  <div class="result-title" style="color:{color}">
                    {result_icon(r['class_name'])} {CLASS_LABELS[r['class_name']].upper()}
                  </div>
                  <div class="result-sub">
                    Confiance : <b>{r['confidence']*100:.1f}%</b> &nbsp;|&nbsp;
                    Sévérité : <b>{sev:.0f}%</b> — {r['severity_label']}
                  </div>
                  <div class="result-sub">{ACTION.get(r['class_name'],'')}</div>
                </div>""", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_fft(signal_raw, fs=fs_input, title="Spectre FFT MCSA"),
                        use_container_width=True
                    )
                with c2:
                    st.plotly_chart(plot_probas(r["all_probas"]), use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lecture fichier : {e}")


# ═════════════════════════════════════════════════════════════
# MODE 4 — COMPARAISON
# ═════════════════════════════════════════════════════════════
elif mode == "📈 Comparaison":
    st.subheader("Comparaison côte à côte de deux défauts")
    st.info(f"🤖 Modèle utilisé : **{model_option.upper()}**")

    c1, c2 = st.columns(2)
    with c1:
        cls_a = st.selectbox("Défaut A", CLASS_NAMES,
                              format_func=lambda x: CLASS_LABELS[x], key="ca")
    with c2:
        cls_b = st.selectbox("Défaut B", CLASS_NAMES, index=4,
                              format_func=lambda x: CLASS_LABELS[x], key="cb")

    if st.button("⚖️ Comparer", type="primary", use_container_width=True):
        from data_engine.signal_generator import SignalGenerator as SG
        g_cmp = SG(DATA_CFG, seed=77)

        with st.spinner("Génération et analyse..."):
            sig_a = g_cmp.generate(make_fault_params(cls_a))
            sig_b = g_cmp.generate(make_fault_params(cls_b))
            r_a   = run_diagnosis_with_model(active_pipeline, sig_a)
            r_b   = run_diagnosis_with_model(active_pipeline, sig_b)

        ca, cb = st.columns(2)
        for col, r, cls, sig in [(ca, r_a, cls_a, sig_a), (cb, r_b, cls_b, sig_b)]:
            sev   = r["severity"]
            color = severity_color(sev)
            with col:
                st.markdown(f"""
                <div class="{result_css_class(sev)}" style="margin-bottom:1rem">
                  <div class="result-title" style="color:{color}">
                    {result_icon(r['class_name'])} {CLASS_LABELS[r['class_name']].upper()}
                  </div>
                  <div class="result-sub">
                    Confiance : {r['confidence']*100:.0f}% &nbsp;|&nbsp;
                    Sévérité : {sev:.0f}%
                  </div>
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(
                    plot_signal(sig, title=CLASS_LABELS[cls]),
                    use_container_width=True
                )
                st.plotly_chart(
                    plot_fft(sig, title="Spectre FFT"),
                    use_container_width=True
                )

        st.markdown("### Comparaison des probabilités")
        labels = [CLASS_LABELS[k] for k in CLASS_NAMES]
        vals_a = [r_a["all_probas"].get(k, 0)*100 for k in CLASS_NAMES]
        vals_b = [r_b["all_probas"].get(k, 0)*100 for k in CLASS_NAMES]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name=CLASS_LABELS[cls_a], x=labels,
                                  y=[float(v) for v in vals_a],
                                  marker_color="#00d4ff"))
        fig_cmp.add_trace(go.Bar(name=CLASS_LABELS[cls_b], x=labels,
                                  y=[float(v) for v in vals_b],
                                  marker_color="#ff6b35"))
        fig_cmp.update_layout(
            **PLOTLY_DARK, barmode="group",
            xaxis=dict(**_AXIS_STYLE, tickangle=-35),
            yaxis=dict(**_AXIS_STYLE, title="Probabilité (%)"),
            height=350,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#5a7090;font-size:0.75rem;letter-spacing:1px'>"
   
    "</div>",
    unsafe_allow_html=True
)