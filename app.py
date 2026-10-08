"""
==============================================================================
SwasthAI — AI-Powered Early Disease Screening Assistant
==============================================================================
Aligned with UN SDG 3: Good Health and Well-being

PURPOSE:
    A Streamlit web application that uses machine learning to screen for
    8 common illnesses based on 20 binary symptoms reported by a user.
    The app generates a fully synthetic dataset inside the code (no external
    files), trains three classifiers, evaluates them, and provides a
    probability-ranked prediction with plain-language self-care advice.

    *** Dataset is synthetic and illustrative, created for educational
        purposes. This tool is NOT a medical diagnosis. ***

ALGORITHMS USED:
    • Random Forest  – an ensemble of many Decision Trees trained on random
      subsets of data and features; predictions are made by majority vote,
      which reduces overfitting and improves accuracy.
    • Decision Tree  – a tree of if/else rules on feature values; easy to
      interpret but can overfit if not depth-limited.
    • Logistic Regression – fits a linear boundary in symptom space and
      converts scores to probabilities via the sigmoid function; fast and
      interpretable but assumes linear separability.

KEY ML CONCEPTS (plain English):
    • Train/test split   – we hold 20 % of data back so the model is
      evaluated on examples it has never seen.
    • Cross-validation   – the training set is split into 5 folds; the model
      trains on 4 and tests on 1, rotating folds; the average score gives a
      more reliable estimate of generalisation.
    • Accuracy           – fraction of predictions that are correct.
    • Precision          – of all predicted positives, how many are truly positive.
    • Recall             – of all true positives, how many did we catch.
    • F1 Score           – harmonic mean of precision and recall; balances both.
    • Feature Importance – how much each symptom reduces impurity across all
      trees in the Random Forest; higher = more informative symptom.

HOW TO RUN:
    pip install streamlit pandas numpy scikit-learn
    streamlit run app.py
==============================================================================
"""

# ── Standard-library imports ──────────────────────────────────────────────────
import warnings  # Suppress harmless convergence warnings from LogisticRegression

# ── Third-party imports ───────────────────────────────────────────────────────
import numpy as np        # Numerical arrays and random sampling
import pandas as pd       # Tabular data manipulation
import streamlit as st    # Web-app framework

# Scikit-learn classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Scikit-learn utilities for splitting, cross-validation and evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Suppress ConvergenceWarning from LogisticRegression on small synthetic data
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – PAGE CONFIGURATION
# Must be the very first Streamlit call in the script.
# ══════════════════════════════════════════════════════════════════════════════

# Set wide layout, browser tab title and favicon icon
st.set_page_config(
    layout="wide",
    page_title="SwasthAI – Disease Screener",
    page_icon="🩺",
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – CUSTOM CSS
# We inject one <style> block to style the whole app.
#
# KEY DESIGN DECISIONS:
# • We only hide #MainMenu and the Streamlit footer — NOT the header element,
#   NOT .stApp, NOT .block-container visibility. Hiding those causes the blank
#   white page bug on Streamlit Cloud.
# • Every colour is specified explicitly (background, text, border) so the app
#   looks correct in both light and dark browser themes.
# • We do NOT use split open/close <div class="card"> across separate
#   st.markdown calls — that breaks Streamlit's layout renderer. All card HTML
#   is self-contained in a single st.markdown call, or we use st.container().
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* ── Import Inter font (loaded by browser, gracefully degrades) ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base font for the whole page ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #0f172a;
    }

    /* ── Hide Streamlit default menu and built-in footer ── */
    /* We intentionally do NOT hide header, .stApp, or .block-container
       because doing so causes the blank-page bug on Streamlit Cloud. */
    #MainMenu          { visibility: hidden; }
    footer             { visibility: hidden; }

    /* ── Page background and top padding ── */
    .stApp             { background-color: #f8fafc; }
    .block-container   { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

    /* ── Hero gradient banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f4c81 0%, #0d9488 100%);
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.4rem;
        color: #ffffff;
    }
    .hero-banner h1 {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        color: #ffffff;
        letter-spacing: -0.4px;
    }
    .hero-banner p {
        font-size: 0.96rem;
        color: rgba(255,255,255,0.88);
        margin: 0 0 1.1rem 0;
    }
    .stat-chips { display: flex; gap: 0.65rem; flex-wrap: wrap; }
    .stat-chip {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 50px;
        padding: 0.22rem 0.85rem;
        font-size: 0.79rem;
        font-weight: 500;
        color: #ffffff;
    }

    /* ── Generic white card used via single-call st.markdown ── */
    .swai-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
        color: #0f172a;
    }
    .swai-card h3 {
        font-size: 1.02rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 0.6rem 0;
    }
    .swai-card p { color: #334155; }

    /* ── Result condition cards ── */
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
        color: #0f172a;
    }
    .result-card.top-result {
        border: 2px solid #0d9488;
        box-shadow: 0 4px 16px rgba(13,148,136,0.14);
    }
    .result-card h4 {
        font-size: 0.97rem;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
        color: #0f172a;
    }
    .result-card p { color: #374151; font-size: 0.83rem; margin: 0; }

    /* ── Confidence badge colours ── */
    .badge {
        display: inline-block;
        border-radius: 50px;
        padding: 0.14rem 0.65rem;
        font-size: 0.74rem;
        font-weight: 600;
    }
    .badge-high   { background: #d1fae5; color: #065f46; }
    .badge-medium { background: #fef3c7; color: #92400e; }
    .badge-low    { background: #fee2e2; color: #991b1b; }

    /* ── Metric stat cards ── */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        padding: 1rem 1rem;
        text-align: center;
        color: #0f172a;
    }
    .metric-card .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0d9488;
    }
    .metric-card .metric-label {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.18rem;
    }

    /* ── Symptom-group small headings ── */
    .symptom-group {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin: 0.9rem 0 0.3rem 0;
    }

    /* ── Risk / urgent callout banners ── */
    .risk-banner {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
        color: #7c2d12;
        font-size: 0.88rem;
    }
    .urgent-banner {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
        color: #7f1d1d;
        font-size: 0.88rem;
    }

    /* ── Notice bar (educational disclaimer) ── */
    .notice-bar {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 9px;
        padding: 0.7rem 1rem;
        margin-bottom: 1.1rem;
        font-size: 0.84rem;
        color: #1e3a8a;
    }

    /* ── Primary action button ── */
    .stButton > button {
        background: linear-gradient(135deg, #0f4c81, #0d9488);
        color: #ffffff !important;
        border: none;
        border-radius: 50px;
        padding: 0.52rem 1.4rem;
        font-weight: 600;
        font-size: 0.88rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.18s ease;
    }
    .stButton > button:hover { opacity: 0.86; }

    /* ── Footer disclaimer ── */
    .footer-disclaimer {
        text-align: center;
        font-size: 0.74rem;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
        padding-top: 0.9rem;
        margin-top: 2rem;
    }

    /* ── Tab bar styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        padding: 0.25rem;
        gap: 0.15rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        font-weight: 500;
        font-size: 0.87rem;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f4c81, #0d9488) !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – CONSTANTS
# Define all symptoms, conditions and per-condition symptom probabilities.
# These are module-level so every function and UI section can reference them.
# ══════════════════════════════════════════════════════════════════════════════

# Ordered list of 20 binary symptom feature names.
# The ORDER here is the training column order — prediction input must match it.
SYMPTOMS: list[str] = [
    "fever", "high_fever", "cough", "sore_throat", "runny_nose",
    "headache", "body_ache", "fatigue", "nausea", "vomiting",
    "diarrhea", "abdominal_pain", "rash", "joint_pain", "chills",
    "loss_of_taste_smell", "shortness_of_breath", "sweating",
    "light_sensitivity", "loss_of_appetite",
]

# The 8 disease conditions the model will predict
CONDITIONS: list[str] = [
    "Common Cold", "Influenza", "Dengue", "Malaria",
    "Typhoid", "Gastroenteritis", "COVID-19-like Illness", "Migraine",
]

# Per-condition symptom probability dictionary.
# Each value is a {symptom: P(symptom=1 | condition)} mapping.
# Symptoms absent from the map get BACKGROUND_PROB (0.05).
# These probabilities are illustrative — NOT derived from clinical data.
CONDITION_SYMPTOM_PROBS: dict[str, dict[str, float]] = {
    "Common Cold": {
        "fever": 0.50, "cough": 0.80, "sore_throat": 0.75,
        "runny_nose": 0.90, "headache": 0.40, "fatigue": 0.50,
        "body_ache": 0.30,
    },
    "Influenza": {
        "fever": 0.85, "high_fever": 0.60, "cough": 0.75,
        "sore_throat": 0.50, "headache": 0.70, "body_ache": 0.85,
        "fatigue": 0.90, "chills": 0.70, "loss_of_appetite": 0.60,
    },
    "Dengue": {
        "high_fever": 0.90, "headache": 0.80, "joint_pain": 0.80,
        "rash": 0.50, "fatigue": 0.70, "nausea": 0.50,
        "loss_of_appetite": 0.60, "body_ache": 0.70,
    },
    "Malaria": {
        "high_fever": 0.80, "chills": 0.90, "sweating": 0.80,
        "headache": 0.70, "nausea": 0.60, "vomiting": 0.50,
        "fatigue": 0.75, "loss_of_appetite": 0.65,
    },
    "Typhoid": {
        "fever": 0.90, "high_fever": 0.60, "headache": 0.65,
        "abdominal_pain": 0.70, "loss_of_appetite": 0.75,
        "fatigue": 0.80, "diarrhea": 0.45, "nausea": 0.50,
    },
    "Gastroenteritis": {
        "nausea": 0.90, "vomiting": 0.85, "diarrhea": 0.90,
        "abdominal_pain": 0.85, "fever": 0.50, "fatigue": 0.60,
        "loss_of_appetite": 0.70,
    },
    "COVID-19-like Illness": {
        "fever": 0.75, "high_fever": 0.45, "cough": 0.70,
        "fatigue": 0.80, "loss_of_taste_smell": 0.75,
        "shortness_of_breath": 0.55, "headache": 0.60,
        "body_ache": 0.60, "loss_of_appetite": 0.55,
    },
    "Migraine": {
        "headache": 0.95, "light_sensitivity": 0.90, "nausea": 0.60,
        "vomiting": 0.35, "fatigue": 0.55,
        # No fever in Migraine — unlisted symptoms default to 0.05
    },
}

# Default probability for any symptom not explicitly listed for a condition
BACKGROUND_PROB: float = 0.05

# How many synthetic patients to generate per condition (8 × 150 = 1 200 total)
PATIENTS_PER_CONDITION: int = 150

# Fraction of labels to randomly reassign (adds realistic noise / overlap)
NOISE_RATE: float = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – CONDITION METADATA (descriptions + advice)
# Plain-English info shown to the user after prediction.
# ══════════════════════════════════════════════════════════════════════════════

CONDITION_INFO: dict[str, dict[str, str]] = {
    "Common Cold": {
        "description": "A mild viral upper respiratory infection causing congestion and sore throat.",
        "self_care": "Rest, stay hydrated, use saline nasal sprays, and try honey-lemon tea.",
        "see_doctor": "If symptoms last more than 10 days, fever exceeds 38.5 °C, or breathing is difficult.",
    },
    "Influenza": {
        "description": "A contagious respiratory illness with abrupt onset of fever, aches and fatigue.",
        "self_care": "Rest at home, drink plenty of fluids, take paracetamol for fever and aches.",
        "see_doctor": "If you have difficulty breathing, chest pain, persistent high fever, or are in a high-risk group.",
    },
    "Dengue": {
        "description": "A mosquito-borne viral fever with severe joint pain, rash and high fever.",
        "self_care": "Rest, take paracetamol (avoid aspirin/ibuprofen), stay well hydrated, use mosquito nets.",
        "see_doctor": "Seek immediate care for bleeding gums, blood in urine/stool, severe abdominal pain or persistent vomiting.",
    },
    "Malaria": {
        "description": "A parasitic disease transmitted by Anopheles mosquitoes causing cyclic fever and chills.",
        "self_care": "Take prescribed antimalarials, rest, stay hydrated, and use insect repellent.",
        "see_doctor": "Consult a doctor immediately — malaria requires prescription antimalarial medication.",
    },
    "Typhoid": {
        "description": "A bacterial infection from contaminated food/water causing prolonged fever and abdominal symptoms.",
        "self_care": "Drink boiled/purified water, eat easily digestible food, maintain strict hand hygiene.",
        "see_doctor": "Typhoid requires antibiotic treatment — consult a doctor promptly if suspected.",
    },
    "Gastroenteritis": {
        "description": "Stomach flu causing nausea, vomiting and diarrhoea, usually viral or food-related.",
        "self_care": "Rehydrate with oral rehydration salts, avoid solid food initially, then reintroduce bland foods.",
        "see_doctor": "If diarrhoea lasts more than 3 days, there is blood in stool, or you have signs of severe dehydration.",
    },
    "COVID-19-like Illness": {
        "description": "A respiratory illness with fever, cough, fatigue and possible loss of taste or smell.",
        "self_care": "Isolate, rest, stay hydrated, monitor oxygen levels, follow local public health guidelines.",
        "see_doctor": "Seek emergency care for shortness of breath, persistent chest pain, confusion or bluish lips.",
    },
    "Migraine": {
        "description": "A neurological condition causing intense recurring headaches, often with light sensitivity and nausea.",
        "self_care": "Rest in a dark quiet room, apply a cold compress, stay hydrated, avoid known triggers.",
        "see_doctor": "If headache is the 'worst of your life', is sudden/thunderclap, or is accompanied by fever and neck stiffness.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – DATA GENERATION (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data   # Run once per session; result is stored in Streamlit's cache
def generate_dataset() -> pd.DataFrame:
    """
    Build a fully synthetic patient dataset.

    For each condition we sample PATIENTS_PER_CONDITION rows by drawing each
    symptom from a Bernoulli(p) distribution where p comes from
    CONDITION_SYMPTOM_PROBS (or BACKGROUND_PROB if not listed).
    After assembling all rows, 5 % of labels are randomly replaced to add
    realistic noise and prevent the dataset from being perfectly separable.

    Returns
    -------
    pd.DataFrame  — 1 200 rows × (20 symptom columns + 'condition' column)
    """
    # Fix the global numpy random seed so the same dataset is produced every run
    np.random.seed(42)

    rows: list[dict] = []

    # Loop over each of the 8 conditions
    for condition in CONDITIONS:
        prob_map = CONDITION_SYMPTOM_PROBS[condition]

        # Generate one row per synthetic patient
        for _ in range(PATIENTS_PER_CONDITION):
            patient: dict[str, object] = {}

            # For each symptom, sample 0 or 1 from a Bernoulli distribution
            for symptom in SYMPTOMS:
                p = prob_map.get(symptom, BACKGROUND_PROB)
                patient[symptom] = int(np.random.binomial(1, p))

            patient["condition"] = condition
            rows.append(patient)

    df = pd.DataFrame(rows)

    # ── 5 % label noise ──────────────────────────────────────────────────────
    # Randomly pick 5 % of rows and swap their condition label for a random one.
    # This prevents perfect separability and simulates real-world ambiguity.
    n_noisy = int(len(df) * NOISE_RATE)
    noisy_idx = np.random.choice(df.index, size=n_noisy, replace=False)
    df.loc[noisy_idx, "condition"] = np.random.choice(CONDITIONS, size=n_noisy)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – MODEL TRAINING & EVALUATION (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource  # Trained model objects are stored once; retrained only on code change
def train_models(df: pd.DataFrame):
    """
    Split data, train three classifiers, compute all metrics and return them.

    WHY THREE MODELS?
    We compare a non-linear ensemble (Random Forest), a single tree (Decision
    Tree) and a linear model (Logistic Regression) so the user can see the
    trade-offs in accuracy and interpretability.

    PANDAS 2/3 COMPATIBILITY:
    pandas 3 uses Arrow-backed dtypes by default. Calling .values on an
    Arrow-backed Series/DataFrame returns an ArrowExtensionArray, NOT a NumPy
    array. train_test_split uses integer scalar indexing that breaks on
    ArrowExtensionArray. We use np.asarray() to force a plain NumPy array
    on both pandas 2.x and 3.x.

    Returns
    -------
    tuple of 11 items:
        rf, dt, lr             – fitted classifier objects
        X_train, X_test        – NumPy feature arrays
        y_train, y_test        – NumPy label arrays
        comparison_df          – model comparison DataFrame
        rf_metrics             – dict of RF accuracy/precision/recall/f1
        conf_df                – confusion matrix as labelled DataFrame
        feat_imp_df            – top-10 feature importances as DataFrame
    """
    # ── Extract features and labels as plain NumPy arrays ────────────────────
    # np.asarray forces a genuine NumPy ndarray regardless of pandas backend.
    # dtype=float ensures scikit-learn's numeric routines work on both old and
    # new pandas. dtype=str forces plain Python strings for the label vector,
    # which scikit-learn's stratified splitter and classifiers expect.
    X = np.asarray(df[SYMPTOMS], dtype=float)   # shape (1200, 20)
    y = np.asarray(df["condition"], dtype=str)  # shape (1200,)

    # Keep feature names as a list; used later for feature importances and
    # for building the prediction input vector in the correct column order.
    feature_names: list[str] = SYMPTOMS  # identical to the module-level list

    # ── Stratified 80/20 train/test split ────────────────────────────────────
    # stratify=y ensures every condition has the same proportion in both sets.
    # random_state=42 makes the split identical on every run.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Instantiate the three classifiers ────────────────────────────────────
    # Random Forest: 200 trees, bootstrap samples, random feature subsets
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    # Decision Tree: capped at depth 8 to reduce overfitting
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    # Logistic Regression: 1000 iterations to ensure convergence
    lr = LogisticRegression(max_iter=1000, random_state=42)

    # ── Train all three on the training set ──────────────────────────────────
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # ── Test-set accuracy for each model ─────────────────────────────────────
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    # ── 5-fold cross-validation mean accuracy (on training data only) ────────
    # This gives a more stable estimate of generalisation than a single split.
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5).mean()
    dt_cv = cross_val_score(dt, X_train, y_train, cv=5).mean()
    lr_cv = cross_val_score(lr, X_train, y_train, cv=5).mean()

    # ── Model comparison table ────────────────────────────────────────────────
    comparison_df = pd.DataFrame({
        "Model": [
            "Random Forest (200 trees)",
            "Decision Tree (depth 8)",
            "Logistic Regression",
        ],
        "Test Accuracy": [f"{rf_acc:.3f}", f"{dt_acc:.3f}", f"{lr_acc:.3f}"],
        "5-Fold CV Mean Accuracy": [f"{rf_cv:.3f}", f"{dt_cv:.3f}", f"{lr_cv:.3f}"],
    })

    # ── Full evaluation metrics for the Random Forest (our final model) ──────
    y_pred_rf = rf.predict(X_test)

    rf_metrics: dict[str, float] = {
        "accuracy":  round(accuracy_score(y_test, y_pred_rf), 4),
        # 'weighted' averages weight each class by its true sample count
        "precision": round(precision_score(y_test, y_pred_rf, average="weighted"), 4),
        "recall":    round(recall_score(y_test, y_pred_rf, average="weighted"), 4),
        "f1":        round(f1_score(y_test, y_pred_rf, average="weighted"), 4),
    }

    # ── Confusion matrix as a labelled DataFrame ──────────────────────────────
    # Rows = true condition, columns = predicted condition.
    # Diagonal = correct predictions; off-diagonal = mistakes.
    cm = confusion_matrix(y_test, y_pred_rf, labels=CONDITIONS)
    conf_df = pd.DataFrame(cm, index=CONDITIONS, columns=CONDITIONS)

    # ── Top-10 feature importances ────────────────────────────────────────────
    # feature_importances_ = mean decrease in node impurity across all 200 trees.
    # Higher value → symptom is a more important predictor.
    importances = rf.feature_importances_
    feat_imp_df = (
        pd.DataFrame({"Symptom": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return (
        rf, dt, lr,
        X_train, X_test, y_train, y_test,
        comparison_df, rf_metrics, conf_df, feat_imp_df,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def badge_class(prob: float) -> str:
    """Return the CSS class name for a confidence badge based on probability."""
    if prob >= 0.50:
        return "badge-high"
    if prob >= 0.25:
        return "badge-medium"
    return "badge-low"


def badge_label(prob: float) -> str:
    """Return a human-readable confidence label for the badge text."""
    if prob >= 0.50:
        return "High confidence"
    if prob >= 0.25:
        return "Moderate"
    return "Low"


def build_symptom_vector(selected: list[str]) -> np.ndarray:
    """
    Convert a list of selected symptom names into a binary NumPy row vector.

    The vector follows the same column order as SYMPTOMS (= training order),
    so the model receives features in exactly the same positions it was trained on.

    Parameters
    ----------
    selected : list of symptom name strings

    Returns
    -------
    np.ndarray of shape (1, 20), dtype float
    """
    # Start with all zeros — unselected symptoms are absent (0)
    vec = np.zeros(len(SYMPTOMS), dtype=float)
    for symptom in selected:
        if symptom in SYMPTOMS:
            # Set the symptom's position in the vector to 1
            vec[SYMPTOMS.index(symptom)] = 1.0
    # reshape to (1, 20) — scikit-learn expects a 2-D array for predict
    return vec.reshape(1, -1)


def safety_check(selected: list[str], illness_days: int) -> list[dict]:
    """
    Apply three safety rules and return a list of warning dicts.

    Each dict has keys 'urgent' (bool) and 'message' (str).

    Rules
    -----
    1. No symptoms selected → advisory to select at least one.
    2. shortness_of_breath selected → urgent red alert.
    3. high_fever AND illness_days >= 5 → amber warning.
    """
    result: list[dict] = []

    # Rule 1: user submitted with no symptoms ticked
    if len(selected) == 0:
        result.append({
            "urgent": False,
            "message": (
                "⚠️ No symptoms selected. "
                "Please choose at least one symptom to get a prediction."
            ),
        })

    # Rule 2: shortness of breath is an emergency red flag in any context
    if "shortness_of_breath" in selected:
        result.append({
            "urgent": True,
            "message": (
                "🚨 Shortness of breath detected. This may indicate a serious "
                "condition. Please seek immediate medical attention or call "
                "emergency services."
            ),
        })

    # Rule 3: sustained high fever (≥ 5 days) needs clinical evaluation
    if "high_fever" in selected and illness_days >= 5:
        result.append({
            "urgent": False,
            "message": (
                "⚠️ High fever lasting 5 or more days is a medical concern. "
                "Please consult a doctor promptly to rule out serious infections."
            ),
        })

    return result


def freq_table_html(freq_df: pd.DataFrame) -> str:
    """
    Build a plain HTML table from a condition × symptom frequency DataFrame.

    Cell background is a blue gradient computed from the cell value (0–1).
    We compute colours in Python so we do NOT need matplotlib's Styler,
    which avoids the matplotlib dependency and the pandas 3 Styler warnings.

    Parameters
    ----------
    freq_df : pd.DataFrame  — rows = conditions, columns = symptoms, values 0–1

    Returns
    -------
    str — complete <table>…</table> HTML string safe for st.markdown
    """
    # Build table header
    col_headers = "".join(
        f'<th style="padding:4px 6px;font-size:0.72rem;'
        f'background:#f1f5f9;color:#334155;white-space:nowrap;">{col}</th>'
        for col in freq_df.columns
    )
    header_row = (
        f'<tr><th style="padding:4px 6px;background:#f1f5f9;color:#334155;">Condition</th>'
        f"{col_headers}</tr>"
    )

    # Build data rows with inline background colour per cell
    data_rows = ""
    for idx, row in freq_df.iterrows():
        cells = ""
        for val in row:
            # Map value 0–1 to a blue hue: white (255) at 0 → #1e40af blue at 1
            intensity = float(val)
            # Interpolate: R 255→30, G 255→64, B 255→175
            r = int(255 - intensity * (255 - 30))
            g = int(255 - intensity * (255 - 64))
            b = int(255 - intensity * (255 - 175))
            text_colour = "#ffffff" if intensity > 0.55 else "#0f172a"
            cells += (
                f'<td style="padding:4px 6px;font-size:0.73rem;text-align:center;'
                f'background:rgb({r},{g},{b});color:{text_colour};">'
                f"{val:.2f}</td>"
            )
        data_rows += (
            f'<tr><td style="padding:4px 6px;font-size:0.73rem;'
            f'font-weight:500;color:#0f172a;white-space:nowrap;">{idx}</td>'
            f"{cells}</tr>"
        )

    return (
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;font-family:Inter,sans-serif;">'
        f"<thead>{header_row}</thead>"
        f"<tbody>{data_rows}</tbody>"
        "</table></div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – DATA LOADING & MODEL TRAINING
# These calls are cached, so they only execute once per deployment session.
# ══════════════════════════════════════════════════════════════════════════════

# Generate the 1 200-row synthetic dataset
df = generate_dataset()

# Train all three models and unpack the 11-item return tuple.
# The tuple order must match train_models()'s return statement exactly.
(
    rf_model, dt_model, lr_model,
    X_train, X_test, y_train, y_test,
    comparison_df, rf_metrics, conf_df, feat_imp_df,
) = train_models(df)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – HERO BANNER & NOTICE BAR
# Rendered once, above the tabs, on every page load.
# ══════════════════════════════════════════════════════════════════════════════

# Gradient hero banner — all HTML is in one self-contained st.markdown call
st.markdown(
    """
    <div class="hero-banner">
        <h1>🩺 SwasthAI</h1>
        <p>AI-Powered Early Disease Screening Assistant &nbsp;·&nbsp;
           Aligned with UN SDG 3: Good Health &amp; Well-being</p>
        <div class="stat-chips">
            <span class="stat-chip">🦠 8 Conditions</span>
            <span class="stat-chip">🔬 20 Symptoms</span>
            <span class="stat-chip">🌲 Random Forest Model</span>
            <span class="stat-chip">📊 1 200 Synthetic Patients</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Educational disclaimer — always visible, one self-contained block
st.markdown(
    """
    <div class="notice-bar">
        📌 <strong>Notice:</strong> Dataset is synthetic and illustrative,
        created for educational purposes.
        <strong>This tool is NOT a medical diagnosis.</strong>
        Always consult a qualified healthcare professional for medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – MAIN TABS
# All page co
