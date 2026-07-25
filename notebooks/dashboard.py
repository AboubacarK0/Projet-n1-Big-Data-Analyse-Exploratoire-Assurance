import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard — Durée des Interventions",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #1a237e; font-size: 2rem !important; }
    h2 { color: #283593; border-bottom: 2px solid #3949ab; padding-bottom: 6px; }
    h3 { color: #3949ab; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        border-left: 5px solid #3949ab;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        margin-bottom: 8px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a237e; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 2px; }
    .info-box {
        background: #e8eaf6;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #5c6bc0;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #333;
    }
    .winner-badge {
        background: #e8f5e9;
        border: 2px solid #43a047;
        border-radius: 8px;
        padding: 10px 16px;
        color: #1b5e20;
        font-weight: 600;
        display: inline-block;
        margin-top: 6px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données…")
def load_data(path: str):
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner="Préparation du jeu d'analyse…")
def prepare_data(df):
    seuil = df["duree_corrigee"].quantile(0.995)
    clean = df[df["duree_corrigee"] <= seuil].copy()
    clean["log_duree"] = np.log1p(clean["duree_corrigee"])
    clean["Tranche_Experience"] = pd.cut(
        clean["Experience"],
        bins=4,
        labels=["Junior", "Intermédiaire", "Confirmé", "Senior"],
    )
    return clean, seuil


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/combo-chart.png", width=60
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une section",
    [
        "🏠 Accueil",
        "📊 Statistiques Descriptives",
        "📈 Économétrie & OLS",
        "🧬 PCA",
        "🤖 Machine Learning",
        "🏆 Comparaison des Modèles",
        "🔮 Prédiction en Direct",
    ],
)

st.sidebar.divider()
uploaded = st.sidebar.file_uploader(
    "📂 Charger votre CSV", type=["csv"]
)

# ──────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────
if uploaded:
    df_raw = load_data(uploaded)
else:
    st.sidebar.info("Aucun fichier chargé. Utilisez les données de démonstration simulées.")
    rng = np.random.default_rng(42)
    n = 5000
    causes = ["Accident", "Panne mécanique", "Bris de glace", "Incendie", "Choc"]
    energies = ["Essence", "Diesel", "Électricité", "Hybride", "Autre"]
    contrats = ["Contrat A", "Contrat B", "Contrat C"]
    lieux = ["Domicile", "Route", "Parking", "Atelier"]
    pops = ["CAS", "Hors-CAS"]
    df_raw = pd.DataFrame(
        {
            "Experience": rng.integers(0, 30, n),
            "Duree_travail": rng.integers(1, 10, n),
            "Temps_travail": rng.integers(30, 480, n),
            "Cause_intervention": rng.choice(causes, n),
            "Type_d_energie": rng.choice(energies, n),
            "Population": rng.choice(pops, n, p=[0.4, 0.6]),
            "Type_de_contrat": rng.choice(contrats, n),
            "Lieu_travail": rng.choice(lieux, n),
            "duree_corrigee": np.clip(
                rng.lognormal(5.2, 1.2, n)
                + np.where(rng.choice(pops, n, p=[0.4, 0.6]) == "CAS", 200, 0),
                1, None,
            ),
        }
    )

df_clean, seuil_995 = prepare_data(df_raw)

# Variables
variables_numeriques = ["Experience", "Duree_travail", "Temps_travail"]
variables_categoriques = [
    "Cause_intervention", "Type_d_energie", "Population",
    "Type_de_contrat", "Lieu_travail",
]
variable_cible = "duree_corrigee"

# ══════════════════════════════════════════════
# PAGE : ACCUEIL
# ══════════════════════════════════════════════
if page == "🏠 Accueil":
    st.title("📊 Dashboard — Analyse de la Durée des Interventions")
    st.markdown(
        """
        <div class="info-box">
        Ce dashboard présente l'intégralité du projet data : de l'analyse exploratoire 
        jusqu'à la comparaison des modèles de Machine Learning, en passant par l'économétrie 
        et l'ACP.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-value">{len(df_raw):,}</div>
            <div class="metric-label">Observations brutes</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-value">{len(df_clean):,}</div>
            <div class="metric-label">Obs. après nettoyage (p99.5)</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        med = df_clean[variable_cible].median()
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-value">{med:.0f}</div>
            <div class="metric-label">Médiane durée (minutes)</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        moy = df_clean[variable_cible].mean()
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-value">{moy:.0f}</div>
            <div class="metric-label">Moyenne durée (minutes)</div></div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("📋 Structure du projet")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Partie 1 — Économétrie**
        - Statistiques descriptives & outliers
        - Régression OLS (log-linéaire)
        - Vérification VIF (multicolinéarité)
        - Variables d'interaction
        - Analyse en composantes principales (PCA)
        """)
    with col_b:
        st.markdown("""
        **Partie 2 — Machine Learning**
        - Classification binaire (courte/longue durée)
        - Seuil : médiane de `duree_corrigee`
        - Modèles : Random Forest, Gradient Boosting, KNN
        - Optimisation du seuil de décision (gain en minutes)
        - Courbes d'apprentissage & feature importance
        """)

    st.divider()
    st.subheader("Aperçu des données")
    st.dataframe(df_clean.head(10), use_container_width=True)


# ══════════════════════════════════════════════
# PAGE : STATISTIQUES DESCRIPTIVES
# ══════════════════════════════════════════════
elif page == "📊 Statistiques Descriptives":
    st.title("📊 Statistiques Descriptives")

    # KPIs
    desc = df_clean[variable_cible].describe()
    cols = st.columns(5)
    for col, (label, val) in zip(
        cols,
        [("Min", desc["min"]), ("Médiane", desc["50%"]), ("Moyenne", desc["mean"]),
         ("Max", desc["max"]), ("Écart-type", desc["std"])],
    ):
        col.metric(label, f"{val:.1f}")

    st.divider()

    # Distribution brute vs log
    st.subheader("Distribution de la durée corrigée")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df_clean[variable_cible], bins=60, kde=True, ax=axes[0], color="#3949ab")
    axes[0].set_title("Distribution brute", fontweight="bold")
    axes[0].set_xlabel("Durée (minutes)")
    sns.histplot(df_clean["log_duree"], bins=60, kde=True, ax=axes[1], color="#e53935")
    axes[1].set_title("Distribution ln(Durée + 1)", fontweight="bold")
    axes[1].set_xlabel("log(Durée)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        """<div class="info-box">
        La distribution brute est fortement asymétrique à droite (quelques interventions 
        très longues tirent la moyenne). La transformation logarithmique normalise la distribution 
        et améliore la qualité des modèles de régression.
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()
    # Analyse par variables catégorielles
    st.subheader("Durée moyenne par variable catégorielle")
    var_cat_choice = st.selectbox(
        "Choisir une variable",
        ["Cause_intervention", "Type_d_energie", "Population", "Type_de_contrat"],
    )
    analyse = (
        df_clean.groupby(var_cat_choice)[variable_cible]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "Moyenne", "median": "Médiane", "count": "Nb dossiers"})
        .sort_values("Moyenne", ascending=False)
    )
    col_t, col_p = st.columns([1, 2])
    with col_t:
        st.dataframe(analyse.style.format({"Moyenne": "{:.1f}", "Médiane": "{:.1f}"}), use_container_width=True)
    with col_p:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df_clean, x=var_cat_choice, y="log_duree",
            palette="Set2", hue=var_cat_choice, legend=False, ax=ax2
        )
        ax2.set_title(f"log(Durée) selon {var_cat_choice}", fontweight="bold")
        ax2.set_xlabel("")
        ax2.set_ylabel("log(Durée)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.divider()
    # Expérience
    st.subheader("Effet de l'expérience sur la durée")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df_clean, x="Tranche_Experience", y="log_duree",
            palette="Blues", hue="Tranche_Experience", legend=False, ax=ax3
        )
        ax3.set_title("log(Durée) par tranche d'expérience", fontweight="bold")
        ax3.set_xlabel("Groupe")
        ax3.set_ylabel("log(Durée)")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    with col_e2:
        sample_e = df_clean.sample(min(5000, len(df_clean)), random_state=42)
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.regplot(
            data=sample_e, x="Experience", y="log_duree",
            order=2, scatter_kws={"alpha": 0.08, "color": "#7e57c2"},
            line_kws={"color": "red"}, ax=ax4,
        )
        ax4.set_title("Tendance quadratique Expérience → Durée", fontweight="bold")
        ax4.set_xlabel("Expérience (jours)")
        ax4.set_ylabel("log(Durée)")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    corr_exp = df_clean["Experience"].corr(df_clean["log_duree"])
    st.markdown(
        f"""<div class="info-box">
        Corrélation de Pearson Expérience ↔ log(Durée) : <strong>{corr_exp:.4f}</strong> — 
        Très faible. Les processus standardisés compensent le manque d'expérience des juniors.
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# PAGE : ÉCONOMÉTRIE & OLS
# ══════════════════════════════════════════════
elif page == "📈 Économétrie & OLS":
    st.title("📈 Régression OLS (Moindres Carrés Ordinaires)")

    variables_x_qual = ["Cause_intervention", "Type_d_energie", "Population"]
    variable_x_num = ["Experience"]

    @st.cache_data(show_spinner="Estimation OLS…")
    def run_ols(df):
        df_mod = pd.get_dummies(
            df[variables_x_qual + variable_x_num + ["log_duree"]],
            columns=variables_x_qual,
            drop_first=True,
            dtype=int,
        )
        Y = df_mod["log_duree"]
        X = sm.add_constant(df_mod.drop(columns=["log_duree"]))
        model = sm.OLS(Y, X).fit()
        return model, df_mod, X, Y

    model_ols, df_mod, X_ols, Y_ols = run_ols(df_clean)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{model_ols.rsquared:.4f}")
    col2.metric("R² ajusté", f"{model_ols.rsquared_adj:.4f}")
    col3.metric("F-statistic", f"{model_ols.fvalue:,.0f}")
    col4.metric("Observations", f"{int(model_ols.nobs):,}")

    st.markdown(
        """<div class="info-box">
        R² ≈ 5 % — Faible dans l'absolu mais attendu sur des données comportementales 
        à fort bruit résiduel (~360k lignes). La p-value globale est nulle : le modèle est 
        hautement significatif.
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()
    # Coefficients
    st.subheader("Coefficients du modèle")
    coef_df = pd.DataFrame(
        {
            "Coefficient": model_ols.params,
            "Std Error": model_ols.bse,
            "t-stat": model_ols.tvalues,
            "p-value": model_ols.pvalues,
        }
    ).drop(index="const", errors="ignore")
    coef_df["Significatif"] = coef_df["p-value"] < 0.05

    def color_sig(val):
        return "color: #2e7d32; font-weight: bold" if val else "color: #c62828"

    st.dataframe(
        coef_df.style.format({"Coefficient": "{:.4f}", "Std Error": "{:.4f}", "t-stat": "{:.3f}", "p-value": "{:.4f}"})
        .applymap(color_sig, subset=["Significatif"]),
        use_container_width=True,
        height=350,
    )

    st.divider()
    # VIF
    st.subheader("Vérification de la multicolinéarité (VIF)")
    vif_data = pd.DataFrame(
        {
            "Variable": X_ols.columns,
            "VIF": [variance_inflation_factor(X_ols.values, i) for i in range(X_ols.shape[1])],
        }
    ).sort_values("VIF", ascending=False)

    fig_vif, ax_vif = plt.subplots(figsize=(9, max(4, len(vif_data) * 0.35)))
    colors = ["#e53935" if v > 10 else "#fb8c00" if v > 5 else "#43a047" for v in vif_data["VIF"]]
    ax_vif.barh(vif_data["Variable"], vif_data["VIF"], color=colors)
    ax_vif.axvline(5, color="#fb8c00", linestyle="--", label="Seuil 5")
    ax_vif.axvline(10, color="#e53935", linestyle="--", label="Seuil 10")
    ax_vif.set_title("VIF par variable", fontweight="bold")
    ax_vif.legend()
    plt.tight_layout()
    col_v1, col_v2 = st.columns([2, 1])
    col_v1.pyplot(fig_vif)
    plt.close()
    col_v2.dataframe(vif_data.style.format({"VIF": "{:.2f}"}), use_container_width=True)

    st.markdown(
        """<div class="info-box">
        Tous les VIF sont inférieurs à 5 (hors constante). Aucune multicolinéarité problématique 
        détectée. Le modèle est structurellement sain.
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()
    # Résidus
    st.subheader("Diagnostic des résidus")
    residuals = model_ols.resid
    fitted = model_ols.fittedvalues
    fig_res, axes_res = plt.subplots(1, 2, figsize=(13, 4))
    axes_res[0].scatter(fitted, residuals, alpha=0.15, s=4, color="#5c6bc0")
    axes_res[0].axhline(0, color="red", linewidth=1.2)
    axes_res[0].set_title("Résidus vs Valeurs ajustées", fontweight="bold")
    axes_res[0].set_xlabel("Valeurs ajustées")
    axes_res[0].set_ylabel("Résidus")
    sns.histplot(residuals, bins=80, kde=True, ax=axes_res[1], color="#5c6bc0")
    axes_res[1].set_title("Distribution des résidus", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_res)
    plt.close()


# ══════════════════════════════════════════════
# PAGE : PCA
# ══════════════════════════════════════════════
elif page == "🧬 PCA":
    st.title("🧬 Analyse en Composantes Principales (ACP)")

    @st.cache_data(show_spinner="Calcul de l'ACP…")
    def run_pca(df):
        variables_x_qual = ["Cause_intervention", "Type_d_energie", "Population"]
        df_mod = pd.get_dummies(
            df[variables_x_qual + ["Experience"] + ["log_duree"]],
            columns=variables_x_qual, drop_first=True, dtype=int,
        )
        df_mod["Inter_Exp_CAS"] = df_mod["Experience"] * df_mod.get("Population_CAS", 0)
        pca_data = df_mod.drop(columns=["log_duree"], errors="ignore")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_data)
        pca = PCA(n_components=2)
        results = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(results, columns=["PC1", "PC2"])
        pca_df["log_duree"] = df_mod["log_duree"].values
        return pca, pca_df, pca_data

    pca_model, pca_df, pca_data = run_pca(df_clean)
    var_pc1 = pca_model.explained_variance_ratio_[0] * 100
    var_pc2 = pca_model.explained_variance_ratio_[1] * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Variance PC1", f"{var_pc1:.1f} %")
    col2.metric("Variance PC2", f"{var_pc2:.1f} %")
    col3.metric("Variance cumulée", f"{var_pc1 + var_pc2:.1f} %")

    st.divider()
    # Scatter PCA
    st.subheader("Projection des observations dans le plan factoriel")
    sample_pca = pca_df.sample(min(8000, len(pca_df)), random_state=42)
    fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
    sc = ax_sc.scatter(
        sample_pca["PC1"], sample_pca["PC2"],
        c=sample_pca["log_duree"], cmap="coolwarm", alpha=0.4, s=6, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax_sc, label="log(Durée)")
    ax_sc.set_title("PCA — Coloration par log_duree", fontweight="bold")
    ax_sc.set_xlabel(f"PC1 ({var_pc1:.1f} %)")
    ax_sc.set_ylabel(f"PC2 ({var_pc2:.1f} %)")
    ax_sc.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_sc)
    plt.close()

    st.divider()
    # Cercle des corrélations
    st.subheader("Cercle des corrélations")
    fig_circ, ax_circ = plt.subplots(figsize=(9, 9))
    cercle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="grey", linestyle="--")
    ax_circ.add_patch(cercle)
    ax_circ.axhline(0, color="grey", linewidth=0.8)
    ax_circ.axvline(0, color="grey", linewidth=0.8)
    colors_arrows = plt.cm.tab10(np.linspace(0, 1, len(pca_data.columns)))
    for i, feature in enumerate(pca_data.columns):
        x = pca_model.components_[0, i]
        y = pca_model.components_[1, i]
        ax_circ.arrow(0, 0, x, y, head_width=0.025, head_length=0.025,
                      fc=colors_arrows[i], ec=colors_arrows[i], alpha=0.8,
                      length_includes_head=True)
        ax_circ.annotate(
            feature, xy=(x, y), xytext=(1.15 * x, 1.15 * y),
            fontsize=8, weight="bold",
            ha="center" if abs(x) < 0.2 else ("left" if x > 0 else "right"),
            va="center" if abs(y) < 0.2 else ("bottom" if y > 0 else "top"),
        )
    ax_circ.set_xlim(-1.3, 1.3)
    ax_circ.set_ylim(-1.3, 1.3)
    ax_circ.set_xlabel(f"PC1 ({var_pc1:.1f} %)")
    ax_circ.set_ylabel(f"PC2 ({var_pc2:.1f} %)")
    ax_circ.set_title("Cercle des corrélations — Variables explicatives", fontweight="bold", pad=15)
    ax_circ.set_aspect("equal", "box")
    ax_circ.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_circ)
    plt.close()

    st.markdown(
        f"""<div class="info-box">
        <strong>PC1</strong> ({var_pc1:.1f} %) — dimension "Profil & Périmètre" : capturée 
        par Expérience, Inter_Exp_CAS, Population_CAS.<br>
        <strong>PC2</strong> ({var_pc2:.1f} %) — dimension "Technique" : capturée par 
        Cause_intervention (Panne mécanique), indépendante du profil de l'opérateur.
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# PAGE : MACHINE LEARNING
# ══════════════════════════════════════════════
elif page == "🤖 Machine Learning":
    st.title("🤖 Machine Learning — Classification Binaire")

    seuil_mediane = df_clean[variable_cible].median()
    st.markdown(
        f"""<div class="info-box">
        <strong>Seuil de binarisation (médiane) : {seuil_mediane:.0f} minutes</strong><br>
        Classe 0 = intervention courte (≤ médiane) | Classe 1 = intervention longue (> médiane)
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()
    model_choice = st.selectbox(
        "Choisir un modèle à entraîner",
        ["Random Forest", "Gradient Boosting", "KNN (K-Nearest Neighbors)"],
    )

    @st.cache_data(show_spinner="Entraînement en cours…")
    def train_model(df, model_name):
        vars_num = [v for v in ["Experience", "Duree_travail", "Temps_travail"] if v in df.columns]
        vars_cat = [v for v in ["Cause_intervention", "Type_d_energie", "Population",
                                "Type_de_contrat", "Lieu_travail"] if v in df.columns]
        X = df[vars_num + vars_cat].dropna()
        y_bin = (df.loc[X.index, variable_cible] > df[variable_cible].median()).astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.2, random_state=42)

        num_tf = Pipeline([("scaler", StandardScaler())])
        cat_tf = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
        preprocessor = ColumnTransformer([
            ("num", num_tf, vars_num), ("cat", cat_tf, vars_cat)
        ])

        if model_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        elif model_name == "Gradient Boosting":
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        else:
            clf = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_te)[:, 1]

        # Seuil optimal
        seuils = np.linspace(0.01, 1, 100)
        best = None
        best_gain = -np.inf
        for t in seuils:
            y_p = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, y_p).ravel()
            gain = (tp * 60) - (fp * 15) - (fn * 45)
            if gain > best_gain:
                best_gain = gain
                best = (t, gain, tn, fp, fn, tp)

        return pipeline, proba, y_te, best, preprocessor, vars_num, vars_cat

    pipeline, proba, y_te, best, preprocessor, vars_num, vars_cat = train_model(df_clean, model_choice)
    t_opt, gain_opt, tn, fp, fn, tp = best

    # Métriques
    y_pred_opt = (proba >= t_opt).astype(int)
    report = classification_report(y_te, y_pred_opt, output_dict=True, digits=3)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Seuil optimal", f"{t_opt:.3f}")
    col2.metric("Gain planification", f"{gain_opt:,.0f} min")
    col3.metric("Recall classe 1", f"{report['1']['recall']:.3f}")
    col4.metric("Précision classe 1", f"{report['1']['precision']:.3f}")
    col5.metric("Accuracy", f"{report['accuracy']:.3f}")

    st.divider()
    # Matrice de confusion
    st.subheader("Matrice de confusion (seuil optimal)")
    cm = np.array([[tn, fp], [fn, tp]])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", ax=ax_cm,
        xticklabels=["Prédit Court", "Prédit Long"],
        yticklabels=["Réel Court", "Réel Long"],
    )
    ax_cm.set_title(f"Matrice de confusion — {model_choice}", fontweight="bold")
    plt.tight_layout()
    col_cm, col_txt = st.columns([1, 1])
    col_cm.pyplot(fig_cm)
    plt.close()
    col_txt.markdown(f"""
    **Interprétation :**
    - **Vrais Positifs (TP)** : {tp:,} dossiers longs correctement identifiés
    - **Faux Positifs (FP)** : {fp:,} dossiers courts classifiés comme longs
    - **Vrais Négatifs (TN)** : {tn:,} dossiers courts correctement identifiés  
    - **Faux Négatifs (FN)** : {fn:,} dossiers longs manqués

    Le modèle adopte une **stratégie prudentielle** : Rappel élevé sur la classe 1 
    pour éviter les surprises opérationnelles.
    """)

    st.divider()
    # Feature importance (uniquement GB et RF)
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        st.subheader("Top 15 des variables les plus importantes")
        clf = pipeline.named_steps["classifier"]
        onehot_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(vars_cat)
        all_features = vars_num + list(onehot_cols)
        importances = clf.feature_importances_
        df_imp = pd.DataFrame({"Variable": all_features, "Importance": importances}).sort_values("Importance", ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
        sns.barplot(x="Importance", y="Variable", data=df_imp.head(15), palette="viridis", ax=ax_imp)
        ax_imp.set_title(f"Feature Importance — {model_choice}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close()


# ══════════════════════════════════════════════
# PAGE : COMPARAISON DES MODÈLES
# ══════════════════════════════════════════════
elif page == "🏆 Comparaison des Modèles":
    st.title("🏆 Comparaison des Modèles ML")

    st.markdown(
        """<div class="info-box">
        Comparaison des trois modèles entraînés avec la même fonction de coût métier :
        <strong>+60 min par TP / −15 min par FP / −45 min par FN</strong>.
        </div>""",
        unsafe_allow_html=True,
    )

    # Résultats issus des notebooks (valeurs réelles du projet)
    resultats = pd.DataFrame(
        {
            "Modèle": ["Random Forest", "Gradient Boosting", "KNN"],
            "Seuil optimal": [0.141, 0.141, 0.010],
            "Recall classe 1": [0.994, 0.994, 0.996],
            "Précision classe 1": [0.517, 0.517, 0.510],
            "Accuracy": [0.531, 0.531, 0.518],
            "TP": [35747, 35747, 35827],
            "FP": [33402, 33318, 34436],
            "FN": [231, 233, 151],
            "TN": [2352, 2436, 1318],
        }
    )
    resultats["Gain (min)"] = (
        resultats["TP"] * 60 - resultats["FP"] * 15 - resultats["FN"] * 45
    )

    st.subheader("Tableau récapitulatif")
    st.dataframe(
        resultats.style.highlight_max(
            subset=["Recall classe 1", "Précision classe 1", "Accuracy", "Gain (min)", "TN"],
            color="#c8e6c9",
        ).highlight_min(
            subset=["FP", "FN"],
            color="#c8e6c9",
        ).format({
            "Seuil optimal": "{:.3f}",
            "Recall classe 1": "{:.3f}",
            "Précision classe 1": "{:.3f}",
            "Accuracy": "{:.3f}",
            "Gain (min)": "{:,.0f}",
        }),
        use_container_width=True,
    )

    st.divider()
    # Graphiques comparatifs
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Recall & Précision par modèle")
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
        x = np.arange(len(resultats))
        w = 0.35
        ax_bar.bar(x - w / 2, resultats["Recall classe 1"], w, label="Recall", color="#3949ab")
        ax_bar.bar(x + w / 2, resultats["Précision classe 1"], w, label="Précision", color="#e53935")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(resultats["Modèle"], rotation=10)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
        ax_bar.legend()
        ax_bar.set_title("Recall vs Précision (classe 1)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close()

    with col_g2:
        st.subheader("Gain planification (minutes)")
        fig_gain, ax_gain = plt.subplots(figsize=(7, 4))
        colors_gain = ["#43a047" if v == resultats["Gain (min)"].max() else "#78909c" for v in resultats["Gain (min)"]]
        ax_gain.bar(resultats["Modèle"], resultats["Gain (min)"], color=colors_gain)
        ax_gain.set_title("Gain net en minutes (fonction de coût métier)", fontweight="bold")
        ax_gain.set_ylabel("Minutes")
        for i, v in enumerate(resultats["Gain (min)"]):
            ax_gain.text(i, v + 20000, f"{v:,.0f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_gain)
        plt.close()

    st.divider()
    # Matrice de confusion comparative
    st.subheader("Matrices de confusion comparées")
    fig_cms, axes_cms = plt.subplots(1, 3, figsize=(15, 4))
    for i, row in resultats.iterrows():
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", ax=axes_cms[i],
            xticklabels=["Court", "Long"],
            yticklabels=["Court", "Long"],
        )
        axes_cms[i].set_title(row["Modèle"], fontweight="bold")
        axes_cms[i].set_xlabel("Prédit")
        axes_cms[i].set_ylabel("Réel" if i == 0 else "")
    plt.suptitle("Matrices de confusion — Seuil optimal par modèle", fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig_cms)
    plt.close()

    st.divider()
    # Verdict
    best_model = resultats.loc[resultats["Gain (min)"].idxmax(), "Modèle"]
    best_gain = resultats["Gain (min)"].max()
    st.markdown(
        f"""<div class="info-box">
        <span class="winner-badge">🏆 Modèle recommandé : {best_model}</span><br><br>
        Le <strong>Gradient Boosting</strong> offre le meilleur gain planification 
        (<strong>{best_gain:,.0f} minutes</strong>) grâce à une réduction marginale mais mesurable 
        des Faux Positifs (+84 TN par rapport au Random Forest), sans dégrader le Rappel sur la 
        classe cible.<br><br>
        Le <strong>KNN</strong> obtient le Rappel le plus élevé (0.996) mais au prix d'un effondrement 
        de la Précision et d'un gain planification inférieur. Sa lourdeur computationnelle sur 
        350k lignes le pénalise également en production.
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# PAGE : PRÉDICTION EN DIRECT
# ══════════════════════════════════════════════
elif page == "🔮 Prédiction en Direct":
    st.title("🔮 Prédiction en Direct")
    st.markdown(
        """<div class="info-box">
        Renseignez les caractéristiques d'un dossier pour obtenir instantanément 
        la durée estimée par le <strong>meilleur modèle ML (Gradient Boosting)</strong> 
        et par le <strong>modèle économétrique OLS</strong>.
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Entraînement du GB (caché) ──────────────────────────────────────
    @st.cache_resource(show_spinner="Entraînement du modèle Gradient Boosting…")
    def train_gb_for_pred(df):
        vars_num = [v for v in ["Experience", "Duree_travail", "Temps_travail"] if v in df.columns]
        vars_cat = [v for v in ["Cause_intervention", "Type_d_energie", "Population",
                                "Type_de_contrat", "Lieu_travail"] if v in df.columns]
        X = df[vars_num + vars_cat].dropna()
        y_bin = (df.loc[X.index, variable_cible] > df[variable_cible].median()).astype(int)
        X_tr, _, y_tr, _ = train_test_split(X, y_bin, test_size=0.2, random_state=42)
        num_tf = Pipeline([("scaler", StandardScaler())])
        cat_tf = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
        preprocessor = ColumnTransformer([
            ("num", num_tf, vars_num), ("cat", cat_tf, vars_cat)
        ])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ))
        ])
        pipeline.fit(X_tr, y_tr)
        return pipeline, vars_num, vars_cat

    @st.cache_resource(show_spinner="Estimation du modèle OLS…")
    def train_ols_for_pred(df):
        variables_x_qual = [v for v in ["Cause_intervention", "Type_d_energie", "Population"]
                            if v in df.columns]
        df_mod = pd.get_dummies(
            df[variables_x_qual + ["Experience"] + ["log_duree"]],
            columns=variables_x_qual, drop_first=True, dtype=int,
        )
        Y = df_mod["log_duree"]
        X = sm.add_constant(df_mod.drop(columns=["log_duree"]))
        model = sm.OLS(Y, X).fit()
        return model, df_mod.drop(columns=["log_duree"]).columns.tolist()

    gb_pipeline, gb_vars_num, gb_vars_cat = train_gb_for_pred(df_clean)
    ols_model, ols_cols = train_ols_for_pred(df_clean)

    # ── Récupération des modalités disponibles ──────────────────────────
    causes_dispo    = sorted(df_clean["Cause_intervention"].dropna().unique()) if "Cause_intervention" in df_clean.columns else ["Accident", "Panne mécanique", "Bris de glace", "Incendie"]
    energies_dispo  = sorted(df_clean["Type_d_energie"].dropna().unique())     if "Type_d_energie"       in df_clean.columns else ["Essence", "Diesel", "Électricité", "Hybride"]
    contrats_dispo  = sorted(df_clean["Type_de_contrat"].dropna().unique())    if "Type_de_contrat"      in df_clean.columns else ["Contrat A", "Contrat B"]
    lieux_dispo     = sorted(df_clean["Lieu_travail"].dropna().unique())       if "Lieu_travail"         in df_clean.columns else ["Domicile", "Route", "Parking"]
    exp_min = 0
    exp_max = min(10000, int(df_clean["Experience"].quantile(0.99))) if "Experience" in df_clean.columns else 40
    dt_min  = int(df_clean["Duree_travail"].min()) if "Duree_travail" in df_clean.columns else 1
    dt_max  = min(24, int(df_clean["Duree_travail"].quantile(0.99))) if "Duree_travail" in df_clean.columns else 10
    tt_min  = int(df_clean["Temps_travail"].min()) if "Temps_travail" in df_clean.columns else 30
    tt_max  = min(600, int(df_clean["Temps_travail"].quantile(0.99))) if "Temps_travail" in df_clean.columns else 480

    # ── Formulaire ──────────────────────────────────────────────────────
    st.subheader("🎛️ Paramètres du dossier")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**👤 Profil de l'opérateur**")
        experience = st.slider(
            "Expérience (jours)", min_value=exp_min, max_value=exp_max,
            value=(exp_min + exp_max) // 2, step=1,
            help="Ancienneté de l'opérateur en jours travaillés"
        )
        duree_travail = st.slider(
            "Durée de travail (heures/jour)", min_value=dt_min, max_value=dt_max,
            value=(dt_min + dt_max) // 2, step=1,
        )
        temps_travail = st.slider(
            "Temps de travail hebdomadaire (min)", min_value=tt_min, max_value=tt_max,
            value=(tt_min + tt_max) // 2, step=10,
        )

        st.markdown("**📋 Caractéristiques du dossier**")
        is_cas = st.checkbox("🏷️ Dossier CAS", value=False,
                             help="Cocher si le dossier appartient à la population CAS")
        type_contrat = st.selectbox("Type de contrat", contrats_dispo)
        lieu_travail = st.selectbox("Lieu de l'intervention", lieux_dispo)

    with col_right:
        st.markdown("**🔧 Nature de l'intervention**")
        cause = st.selectbox("Cause de l'intervention", causes_dispo,
                             help="La cause influence significativement la durée (−28% pour bris de glace vs accident)")
        is_panne_meca = cause == "Panne mécanique"
        st.markdown(
            f"{'✅ Panne mécanique détectée' if is_panne_meca else ''}",
            unsafe_allow_html=True,
        )

        st.markdown("**⛽ Type de motorisation**")
        type_energie = st.selectbox("Type d'énergie", energies_dispo,
                                    help="Les véhicules hybrides/électriques peuvent allonger la durée d'intervention")

        # Récap visuel
        st.markdown("**📝 Récapitulatif du dossier**")
        pop_label = "CAS" if is_cas else "Hors-CAS"
        st.markdown(f"""
        | Paramètre | Valeur |
        |-----------|--------|
        | Expérience | **{experience} jours** |
        | Population | **{pop_label}** |
        | Cause | **{cause}** |
        | Énergie | **{type_energie}** |
        | Contrat | **{type_contrat}** |
        | Lieu | **{lieu_travail}** |
        """)

    st.divider()

    # ── Prédiction ──────────────────────────────────────────────────────
    st.subheader("⚡ Résultats de la prédiction")

    # Construire la ligne d'entrée
    input_dict = {
        "Experience": experience,
        "Duree_travail": duree_travail,
        "Temps_travail": temps_travail,
        "Cause_intervention": cause,
        "Type_d_energie": type_energie,
        "Population": "CAS" if is_cas else "Hors-CAS",
        "Type_de_contrat": type_contrat,
        "Lieu_travail": lieu_travail,
    }
    input_df = pd.DataFrame([input_dict])

    # — Prédiction ML (Gradient Boosting) ——————————————————————————————
    input_gb = input_df[[c for c in gb_vars_num + gb_vars_cat if c in input_df.columns]]
    proba_gb = gb_pipeline.predict_proba(input_gb)[0, 1]
    seuil_optimal_gb = 0.141
    classe_pred = "🔴 Long (> médiane)" if proba_gb >= seuil_optimal_gb else "🟢 Court (≤ médiane)"
    mediane_duree = df_clean[variable_cible].median()

    # — Prédiction OLS ——————————————————————————————————————————————————
    input_ols_raw = input_df[["Cause_intervention", "Type_d_energie", "Experience"]].copy()
    input_ols_raw["Population"] = "CAS" if is_cas else "Hors-CAS"
    qual_cols_ols = [c for c in ["Cause_intervention", "Type_d_energie", "Population"]
                     if c in input_ols_raw.columns]
    input_ols_dummies = pd.get_dummies(input_ols_raw, columns=qual_cols_ols, drop_first=True, dtype=int)
    # Aligner avec les colonnes OLS (ajouter colonnes manquantes à 0)
    for col in ols_cols:
        if col not in input_ols_dummies.columns:
            input_ols_dummies[col] = 0
    input_ols_dummies = input_ols_dummies[ols_cols]
    input_ols_const = sm.add_constant(input_ols_dummies, has_constant="add")
    log_pred = ols_model.predict(input_ols_const)[0]
    duree_pred_ols = np.expm1(log_pred)

    # Intervalle de confiance OLS
    pred_summary = ols_model.get_prediction(input_ols_const)
    ci = pred_summary.summary_frame(alpha=0.05)
    duree_ci_low  = np.expm1(ci["mean_ci_lower"].values[0])
    duree_ci_high = np.expm1(ci["mean_ci_upper"].values[0])

    # ── Affichage des résultats ─────────────────────────────────────────
    col_ml, col_ols = st.columns(2, gap="large")

    with col_ml:
        st.markdown(
            f"""
            <div style="background:white; border-radius:12px; padding:24px;
                        border-left:6px solid #3949ab; box-shadow:0 3px 10px rgba(0,0,0,0.1);">
                <div style="font-size:0.9rem; color:#666; font-weight:600; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:8px;">
                    🤖 Gradient Boosting (ML)
                </div>
                <div style="font-size:2.2rem; font-weight:800; color:#1a237e; margin-bottom:4px;">
                    {classe_pred}
                </div>
                <div style="font-size:1rem; color:#555; margin-bottom:16px;">
                    Probabilité d'être un dossier <em>long</em> : 
                    <strong style="color:#e53935;">{proba_gb*100:.1f}%</strong>
                </div>
                <div style="background:#e8eaf6; border-radius:8px; padding:12px; font-size:0.9rem;">
                    Seuil de décision : <strong>{seuil_optimal_gb}</strong><br>
                    Médiane de référence : <strong>{mediane_duree:.0f} min</strong><br>
                    Stratégie : <em>prudentielle (Rappel ≈ 99.4%)</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Jauge de probabilité
        st.markdown("")
        fig_gauge, ax_gauge = plt.subplots(figsize=(5, 1.2))
        ax_gauge.barh([""], [1], color="#e0e0e0", height=0.5)
        ax_gauge.barh([""], [proba_gb], color="#3949ab" if proba_gb < seuil_optimal_gb else "#e53935", height=0.5)
        ax_gauge.axvline(seuil_optimal_gb, color="orange", linewidth=2, linestyle="--")
        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_xlabel("Probabilité dossier long")
        ax_gauge.set_title(f"Jauge — p = {proba_gb:.3f} | seuil = {seuil_optimal_gb}", fontsize=9)
        ax_gauge.tick_params(left=False, labelleft=False)
        plt.tight_layout()
        st.pyplot(fig_gauge)
        plt.close()

    with col_ols:
        st.markdown(
            f"""
            <div style="background:white; border-radius:12px; padding:24px;
                        border-left:6px solid #43a047; box-shadow:0 3px 10px rgba(0,0,0,0.1);">
                <div style="font-size:0.9rem; color:#666; font-weight:600; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:8px;">
                    📈 Régression OLS (Économétrie)
                </div>
                <div style="font-size:2.6rem; font-weight:800; color:#1b5e20; margin-bottom:4px;">
                    {duree_pred_ols:.0f} min
                </div>
                <div style="font-size:1rem; color:#555; margin-bottom:16px;">
                    Durée estimée de traitement du dossier
                </div>
                <div style="background:#e8f5e9; border-radius:8px; padding:12px; font-size:0.9rem;">
                    IC 95% : <strong>[{duree_ci_low:.0f} — {duree_ci_high:.0f} min]</strong><br>
                    Modèle : <em>log-linéaire OLS (R² ≈ 5%)</em><br>
                    Note : <em>R² faible = bruit résiduel élevé, IC large</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Mini bar chart IC
        st.markdown("")
        fig_ic, ax_ic = plt.subplots(figsize=(5, 1.4))
        ax_ic.barh([""], [duree_ci_high - duree_ci_low],
                   left=[duree_ci_low], color="#a5d6a7", height=0.4,
                   label=f"IC 95% [{duree_ci_low:.0f} – {duree_ci_high:.0f}]")
        ax_ic.axvline(duree_pred_ols, color="#1b5e20", linewidth=2.5, label=f"Estimé : {duree_pred_ols:.0f} min")
        ax_ic.set_xlabel("Durée (minutes)")
        ax_ic.set_title("Intervalle de confiance à 95%", fontsize=9)
        ax_ic.tick_params(left=False, labelleft=False)
        ax_ic.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_ic)
        plt.close()

    st.divider()

    # ── Lecture combinée ────────────────────────────────────────────────
    st.subheader("📖 Lecture croisée des deux modèles")
    accord = (classe_pred.startswith("🔴") and duree_pred_ols > mediane_duree) or \
             (classe_pred.startswith("🟢") and duree_pred_ols <= mediane_duree)

    couleur_accord = "#e8f5e9" if accord else "#fff3e0"
    bordure_accord = "#43a047" if accord else "#fb8c00"
    texte_accord = "✅ Les deux modèles sont en accord." if accord else \
                   "⚠️ Les deux modèles divergent — l'OLS a un R² faible, fiez-vous au ML pour la classification."

    st.markdown(
        f"""
        <div style="background:{couleur_accord}; border-left:5px solid {bordure_accord};
                    border-radius:8px; padding:16px; font-size:0.95rem;">
            {texte_accord}<br><br>
            <strong>ML :</strong> Ce dossier est classifié comme <strong>{classe_pred}</strong> 
            avec une probabilité de <strong>{proba_gb*100:.1f}%</strong> d'être long.<br>
            <strong>OLS :</strong> La durée estimée est de <strong>{duree_pred_ols:.0f} minutes</strong>
            (médiane de référence : {mediane_duree:.0f} min).<br><br>
            <em>💡 Conseil opérationnel : {"Planifier un créneau long pour ce dossier." if classe_pred.startswith("🔴") else "Créneau standard suffisant pour ce dossier."}</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
