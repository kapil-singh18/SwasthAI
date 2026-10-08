"""
==============================================================================
AI Symptom Checker: Early Disease Screening Assistant
==============================================================================
Aligned with UN SDG 3: Good Health and Well-being

PURPOSE:
    A Streamlit web application that uses machine learning to screen for
    8 common illnesses based on 20 binary symptoms reported by a user.
    The app generates a synthetic dataset, trains three classifiers, evaluates
    them, and provides a probability-ranked prediction with self-care advice.

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
import numpy as np                        # Numerical arrays and random sampling
import pandas as pd                       # Tabular data manipulation
import streamlit as st                    # Web-app framework
from sklearn.ensemble import RandomForestClassifier      # Ensemble classifier
from sklearn.tree import DecisionTreeClassifier          # Single-tree classifier
from sklearn.linear_model import LogisticRegression      # Linear classifier
from sklearn.model_selection import train_test_split, cross_val_score  # Data splitting & CV
from sklearn.metrics import (
    accuracy_score,          # Overall fraction of correct predictions
    precision_score,         # Weighted precision across all classes
    recall_score,            # Weighted recall across all classes
    f1_score,                # Weighted F1 across all classes
    confusion_matrix,        # Matrix of true vs predicted labels
)

# Suppress ConvergenceWarning from LogisticRegression on small synthetic data
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – PAGE CONFIGURATION & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

# Configure the Streamlit page: wide layout, custom title and favicon
st.set_page_config(
    layout="wide",
    page_title="AI Disease Predictor",
    page_icon="🩺",
)

# Inject custom CSS to create a professional SaaS-health-app visual style.
# We hide default Streamlit chrome (menu bar, footer, header) and apply our
# own card, typography, button and colour-scheme rules.
st.markdown(
    """
    <style>
    /* ── Import Inter font from Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Reset and base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ── Hide default Streamlit UI chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 0rem; padding-bottom: 2rem;}

    /* ── Soft off-white page background ── */
    .stApp {
        background: #f5f7fa;
    }

    /* ── Hero gradient banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f4c81 0%, #1a7a8a 60%, #22a8b0 100%);
        border-radius: 20px;
        padding: 2.5rem 2.8rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-banner h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        font-size: 1rem;
        opacity: 0.88;
        margin: 0 0 1.2rem 0;
    }
    .stat-chips {display: flex; gap: 0.75rem; flex-wrap: wrap;}
    .stat-chip {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 50px;
        padding: 0.25rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* ── Generic white rounded card ── */
    .card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e8edf2;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .card h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a2636;
        margin-top: 0;
    }

    /* ── Result condition cards ── */
    .result-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e8edf2;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.9rem;
    }
    .result-card.top-result {
        border: 2px solid #1a7a8a;
        box-shadow: 0 4px 18px rgba(26,122,138,0.15);
    }
    .result-card h4 {font-size: 1rem; font-weight: 600; margin: 0 0 0.3rem 0; color: #1a2636;}

    /* ── Confidence badge colours ── */
    .badge {
        display: inline-block;
        border-radius: 50px;
        padding: 0.15rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-high   {background: #d1fae5; color: #065f46;}
    .badge-medium {background: #fef3c7; color: #92400e;}
    .badge-low    {background: #fee2e2; color: #991b1b;}

    /* ── Stat metric cards ── */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e8edf2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f4c81;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #6b7a8d;
        margin-top: 0.2rem;
    }

    /* ── Symptom group headings ── */
    .symptom-group {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7a8d;
        margin: 1rem 0 0.4rem 0;
    }

    /* ── Warning / risk callout ── */
    .risk-banner {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: #7c2d12;
        font-size: 0.9rem;
    }
    .urgent-banner {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: #7f1d1d;
        font-size: 0.9rem;
    }

    /* ── Primary button styling via targeting Streamlit's button ── */
    .stButton > button {
        background: linear-gradient(135deg, #0f4c81, #1a7a8a);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.55rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s ease;
    }
    .stButton > button:hover {
        opacity: 0.88;
        color: white;
    }

    /* ── Footer disclaimer ── */
    .footer-disclaimer {
        text-align: center;
        font-size: 0.75rem;
        color: #9ca3af;
        border-top: 1px solid #e8edf2;
        padding-top: 1rem;
        margin-top: 2rem;
    }

    /* ── Tab overrides ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e8edf2;
        padding: 0.3rem;
        gap: 0.2rem;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.88rem;
        color: #6b7a8d;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f4c81, #1a7a8a) !important;
        color: white !important;
    }

    /* ── Dataframe / table ── */
    .dataframe thead th {
        background-color: #f5f7fa !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – SYNTHETIC DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# All 20 binary symptom feature names used throughout the app
SYMPTOMS = [
    "fever", "high_fever", "cough", "sore_throat", "runny_nose",
    "headache", "body_ache", "fatigue", "nausea", "vomiting",
    "diarrhea", "abdominal_pain", "rash", "joint_pain", "chills",
    "loss_of_taste_smell", "shortness_of_breath", "sweating",
    "light_sensitivity", "loss_of_appetite",
]

# The 8 disease conditions the model will predict
CONDITIONS = [
    "Common Cold", "Influenza", "Dengue", "Malaria",
    "Typhoid", "Gastroenteritis", "COVID-19-like Illness", "Migraine",
]

# Per-condition symptom probability dictionary.
# Each key is a condition; the value is a dict of {symptom: probability_of_1}.
# Symptoms not listed default to a low background probability (0.05),
# reflecting that any symptom can occasionally appear in any illness.
CONDITION_SYMPTOM_PROBS: dict[str, dict[str, float]] = {
    "Common Cold": {
        "fever": 0.5, "cough": 0.8, "sore_throat": 0.75,
        "runny_nose": 0.9, "headache": 0.4, "fatigue": 0.5,
        "body_ache": 0.3,
    },
    "Influenza": {
        "fever": 0.85, "high_fever": 0.6, "cough": 0.75,
        "sore_throat": 0.5, "headache": 0.7, "body_ache": 0.85,
        "fatigue": 0.9, "chills": 0.7, "loss_of_appetite": 0.6,
    },
    "Dengue": {
        "high_fever": 0.9, "headache": 0.8, "joint_pain": 0.8,
        "rash": 0.5, "fatigue": 0.7, "nausea": 0.5,
        "loss_of_appetite": 0.6, "body_ache": 0.7,
    },
    "Malaria": {
        "high_fever": 0.8, "chills": 0.9, "sweating": 0.8,
        "headache": 0.7, "nausea": 0.6, "vomiting": 0.5,
        "fatigue": 0.75, "loss_of_appetite": 0.65,
    },
    "Typhoid": {
        "fever": 0.9, "high_fever": 0.6, "headache": 0.65,
        "abdominal_pain": 0.7, "loss_of_appetite": 0.75,
        "fatigue": 0.8, "diarrhea": 0.45, "nausea": 0.5,
    },
    "Gastroenteritis": {
        "nausea": 0.9, "vomiting": 0.85, "diarrhea": 0.9,
        "abdominal_pain": 0.85, "fever": 0.5, "fatigue": 0.6,
        "loss_of_appetite": 0.7,
    },
    "COVID-19-like Illness": {
        "fever": 0.75, "high_fever": 0.45, "cough": 0.7,
        "fatigue": 0.8, "loss_of_taste_smell": 0.75,
        "shortness_of_breath": 0.55, "headache": 0.6,
        "body_ache": 0.6, "loss_of_appetite": 0.55,
    },
    "Migraine": {
        "headache": 0.95, "light_sensitivity": 0.9, "nausea": 0.6,
        "vomiting": 0.35, "fatigue": 0.55,
        # Migraine has no fever, so defaults keep fever very low
    },
}

# Background probability for any symptom not explicitly listed in a condition
BACKGROUND_PROB = 0.05

# Number of synthetic patients to generate per condition
PATIENTS_PER_CONDITION = 150

# Noise rate: 5 % of labels are randomly flipped to prevent perfect separation
NOISE_RATE = 0.05


@st.cache_data  # Cache the generated dataset so it is only created once per session
def generate_dataset() -> pd.DataFrame:
    """
    Generate a synthetic patient dataset with binary symptom features.

    For each condition and each symptom, we draw a Bernoulli sample
    (0 or 1) using the condition-specific probability.  After stacking
    all conditions we flip 5 % of labels at random to add noise.

    Returns
    -------
    pd.DataFrame
        Rows = patients, columns = symptoms + 'condition' label.
    """
    # Fix the random seed so results are reproducible across runs
    np.random.seed(42)

    rows = []  # Accumulate one row (dict) per simulated patient

    # Iterate over each disease condition
    for condition in CONDITIONS:
        # Retrieve this condition's symptom-probability mapping
        prob_map = CONDITION_SYMPTOM_PROBS[condition]

        # Generate PATIENTS_PER_CONDITION rows for this condition
        for _ in range(PATIENTS_PER_CONDITION):
            patient: dict = {}  # Single patient's symptom vector

            # For each of the 20 symptoms, sample a 0 or 1
            for symptom in SYMPTOMS:
                # Use condition-specific probability if defined, else background
                p = prob_map.get(symptom, BACKGROUND_PROB)
                # np.random.binomial(1, p) gives 1 with probability p
                patient[symptom] = int(np.random.binomial(1, p))

            # Store the ground-truth condition label
            patient["condition"] = condition
            rows.append(patient)

    # Build a DataFrame from the list of patient dicts
    df = pd.DataFrame(rows)

    # ── Add 5 % label-independent noise ──────────────────────────────────────
    # Randomly select 5 % of rows and assign them a random (possibly different)
    # condition label.  This simulates real-world ambiguity and overlap.
    n_noisy = int(len(df) * NOISE_RATE)
    # Choose random row indices to corrupt
    noisy_indices = np.random.choice(df.index, size=n_noisy, replace=False)
    # Assign a uniformly random condition to each noisy row
    df.loc[noisy_indices, "condition"] = np.random.choice(CONDITIONS, size=n_noisy)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – MODEL TRAINING & EVALUATION (cached so it runs once)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource  # Cache the trained models and metrics; retrain only when code changes
def train_models(df: pd.DataFrame):
    """
    Split data, train three classifiers, compute metrics, and return everything.

    Steps
    -----
    1. Separate features (X) and labels (y).
    2. Stratified 80/20 train-test split.
    3. Fit RandomForest, DecisionTree and LogisticRegression.
    4. Compute test accuracy and 5-fold CV mean accuracy for each.
    5. Compute full metrics for the final Random Forest model.

    Returns
    -------
    tuple: (rf, dt, lr, X_train, X_test, y_train, y_test,
            comparison_df, rf_metrics, conf_df, feat_imp_df)
    """
    # ── Features and labels ──────────────────────────────────────────────────
    X = df[SYMPTOMS].values   # 2-D array: rows = patients, cols = symptoms
    y = df["condition"].values  # 1-D array: disease label per patient

    # ── Stratified train/test split (80 % train, 20 % test) ─────────────────
    # stratify=y ensures each condition is proportionally represented in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Define the three classifiers ─────────────────────────────────────────
    # Random Forest: 200 trees, each trained on a random bootstrap sample
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Decision Tree: limited to depth 8 to reduce overfitting
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)

    # Logistic Regression: increases max iterations to ensure convergence
    lr = LogisticRegression(max_iter=1000, random_state=42)

    # ── Train all three models on the training set ───────────────────────────
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # ── Compute test-set accuracy for each model ──────────────────────────────
    rf_acc  = accuracy_score(y_test, rf.predict(X_test))
    dt_acc  = accuracy_score(y_test, dt.predict(X_test))
    lr_acc  = accuracy_score(y_test, lr.predict(X_test))

    # ── 5-fold cross-validation on the training set ───────────────────────────
    # cv=5 means the data is split into 5 equal parts; we train on 4, test on 1,
    # rotating 5 times and averaging the scores for a robust estimate.
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5).mean()
    dt_cv = cross_val_score(dt, X_train, y_train, cv=5).mean()
    lr_cv = cross_val_score(lr, X_train, y_train, cv=5).mean()

    # ── Model comparison table ────────────────────────────────────────────────
    comparison_df = pd.DataFrame({
        "Model": ["Random Forest (200 trees)", "Decision Tree (depth 8)", "Logistic Regression"],
        "Test Accuracy": [f"{rf_acc:.3f}", f"{dt_acc:.3f}", f"{lr_acc:.3f}"],
        "5-Fold CV Mean Accuracy": [f"{rf_cv:.3f}", f"{dt_cv:.3f}", f"{lr_cv:.3f}"],
    })

    # ── Full metrics for the final Random Forest model ───────────────────────
    y_pred_rf = rf.predict(X_test)

    rf_metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred_rf), 4),
        # weighted averages weight each class by its support (number of true samples)
        "precision": round(precision_score(y_test, y_pred_rf, average="weighted"), 4),
        "recall":    round(recall_score(y_test, y_pred_rf, average="weighted"), 4),
        "f1":        round(f1_score(y_test, y_pred_rf, average="weighted"), 4),
    }

    # ── Confusion matrix as a labelled DataFrame ──────────────────────────────
    # Rows = true labels, columns = predicted labels; diagonal = correct predictions
    cm = confusion_matrix(y_test, y_pred_rf, labels=CONDITIONS)
    conf_df = pd.DataFrame(cm, index=CONDITIONS, columns=CONDITIONS)

    # ── Top 10 feature importances ────────────────────────────────────────────
    # feature_importances_ gives the mean decrease in impurity for each symptom
    importances = rf.feature_importances_
    feat_imp_df = (
        pd.DataFrame({"Symptom": SYMPTOMS, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return (rf, dt, lr, X_train, X_test, y_train, y_test,
            comparison_df, rf_metrics, conf_df, feat_imp_df)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – CONDITION METADATA (descriptions, advice)
# ══════════════════════════════════════════════════════════════════════════════

# Plain-English description and self-care advice for each condition.
# Displayed in the prediction results to give actionable guidance.
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
        "see_doctor": "Seek immediate care for bleeding gums, blood in urine/stool, sudden severe abdominal pain, or persistent vomiting.",
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
        "self_care": "Rehydrate with oral rehydration salts, avoid solid food initially, gradually reintroduce bland foods.",
        "see_doctor": "If diarrhoea lasts more than 3 days, there is blood in stool, or you show signs of severe dehydration.",
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
# SECTION 5 – HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def badge_class(prob: float) -> str:
    """Return CSS badge class based on probability threshold."""
    if prob >= 0.50:
        return "badge-high"
    elif prob >= 0.25:
        return "badge-medium"
    return "badge-low"


def badge_label(prob: float) -> str:
    """Return human-readable confidence label."""
    if prob >= 0.50:
        return "High confidence"
    elif prob >= 0.25:
        return "Moderate"
    return "Low"


def build_symptom_vector(selected_symptoms: list[str]) -> np.ndarray:
    """Convert a list of selected symptom names into a binary feature vector."""
    # Create a zero vector of length equal to the number of symptoms
    vec = np.zeros(len(SYMPTOMS), dtype=int)
    # Set the position of each selected symptom to 1
    for symptom in selected_symptoms:
        if symptom in SYMPTOMS:
            vec[SYMPTOMS.index(symptom)] = 1
    return vec.reshape(1, -1)  # Reshape to 2-D for scikit-learn predict


def safety_check(selected_symptoms: list[str], high_fever_days: int) -> list[str]:
    """
    Apply safety rules and return a list of warning messages.

    Rules
    -----
    • No symptoms selected → prompt user to select at least one.
    • shortness_of_breath selected → urgent care warning.
    • high_fever selected AND days ≥ 5 → prolonged fever warning.
    """
    warnings_list = []

    # Rule 1: user has not selected any symptom
    if len(selected_symptoms) == 0:
        warnings_list.append(
            "⚠️ No symptoms selected. Please choose at least one symptom to proceed."
        )

    # Rule 2: shortness of breath is an emergency red flag
    if "shortness_of_breath" in selected_symptoms:
        warnings_list.append(
            "🚨 **Shortness of breath detected.** This may indicate a serious condition. "
            "Please seek immediate medical attention or call emergency services."
        )

    # Rule 3: high fever lasting more than 5 days requires clinical evaluation
    if "high_fever" in selected_symptoms and high_fever_days >= 5:
        warnings_list.append(
            "⚠️ **High fever lasting 5 or more days** is a medical concern. "
            "Please consult a doctor promptly to rule out serious infections."
        )

    return warnings_list


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – DATA LOADING & MODEL TRAINING (executed once, cached)
# ══════════════════════════════════════════════════════════════════════════════

# Generate the synthetic dataset (cached after first call)
df = generate_dataset()

# Train all models and unpack the returned tuple
(rf_model, dt_model, lr_model,
 X_train, X_test, y_train, y_test,
 comparison_df, rf_metrics, conf_df, feat_imp_df) = train_models(df)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════

# Render the gradient hero section at the very top of the page
st.markdown(
    """
    <div class="hero-banner">
        <h1>🩺 AI Symptom Checker</h1>
        <p>Early Disease Screening Assistant &nbsp;·&nbsp; Aligned with UN SDG 3: Good Health &amp; Well-being</p>
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

# ── Educational disclaimer notice (always visible) ────────────────────────────
st.markdown(
    """
    <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
                padding:0.75rem 1rem; margin-bottom:1.2rem; font-size:0.85rem; color:#1e40af;">
        📌 <strong>Notice:</strong> Dataset is synthetic and illustrative, created for
        educational purposes. <strong>This tool is NOT a medical diagnosis.</strong>
        Always consult a qualified healthcare professional for medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

# Create four navigation tabs for the app
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Symptom Checker", "📈 Model Performance", "📊 Data Insights", "ℹ️ About Project"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – SYMPTOM CHECKER
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # Two-column layout: left for inputs, right for results
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        # ── Symptom input card ────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🩹 Select Your Symptoms")
        st.markdown(
            '<p style="font-size:0.83rem;color:#6b7a8d;margin-top:-0.5rem;">'
            "Check all symptoms you are currently experiencing.</p>",
            unsafe_allow_html=True,
        )

        # ── Group 1: General symptoms ─────────────────────────────────────────
        st.markdown('<p class="symptom-group">🌡️ General</p>', unsafe_allow_html=True)
        # Arrange checkboxes in 3 columns for compact layout
        g1_col1, g1_col2, g1_col3 = st.columns(3)
        sel_fever        = g1_col1.checkbox("Fever")
        sel_high_fever   = g1_col2.checkbox("High Fever")
        sel_chills       = g1_col3.checkbox("Chills")
        sel_sweating     = g1_col1.checkbox("Sweating")
        sel_fatigue      = g1_col2.checkbox("Fatigue")
        sel_body_ache    = g1_col3.checkbox("Body Ache")
        sel_headache     = g1_col1.checkbox("Headache")
        sel_loss_app     = g1_col2.checkbox("Loss of Appetite")

        # ── Group 2: Respiratory symptoms ────────────────────────────────────
        st.markdown('<p class="symptom-group">🫁 Respiratory</p>', unsafe_allow_html=True)
        g2_col1, g2_col2, g2_col3 = st.columns(3)
        sel_cough        = g2_col1.checkbox("Cough")
        sel_sore_throat  = g2_col2.checkbox("Sore Throat")
        sel_runny_nose   = g2_col3.checkbox("Runny Nose")
        sel_sob          = g2_col1.checkbox("Shortness of Breath")
        sel_loss_ts      = g2_col2.checkbox("Loss of Taste/Smell")

        # ── Group 3: Digestive symptoms ───────────────────────────────────────
        st.markdown('<p class="symptom-group">🫃 Digestive</p>', unsafe_allow_html=True)
        g3_col1, g3_col2, g3_col3 = st.columns(3)
        sel_nausea       = g3_col1.checkbox("Nausea")
        sel_vomiting     = g3_col2.checkbox("Vomiting")
        sel_diarrhea     = g3_col3.checkbox("Diarrhoea")
        sel_abdom_pain   = g3_col1.checkbox("Abdominal Pain")

        # ── Group 4: Other symptoms ───────────────────────────────────────────
        st.markdown('<p class="symptom-group">🔎 Other</p>', unsafe_allow_html=True)
        g4_col1, g4_col2, g4_col3 = st.columns(3)
        sel_rash         = g4_col1.checkbox("Rash")
        sel_joint_pain   = g4_col2.checkbox("Joint Pain")
        sel_light_sens   = g4_col3.checkbox("Light Sensitivity")

        st.markdown("</div>", unsafe_allow_html=True)  # Close symptom card

        # ── Patient context card ──────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 👤 Patient Context")

        # Age slider: range 1–100
        age = st.slider("Age (years)", min_value=1, max_value=100, value=30, step=1)

        # Days-of-illness slider: 1–30 days
        illness_days = st.slider("Days of illness", min_value=1, max_value=30, value=3, step=1)

        # Map checkbox states back to the canonical symptom-name strings
        selected_symptoms: list[str] = []
        if sel_fever:         selected_symptoms.append("fever")
        if sel_high_fever:    selected_symptoms.append("high_fever")
        if sel_cough:         selected_symptoms.append("cough")
        if sel_sore_throat:   selected_symptoms.append("sore_throat")
        if sel_runny_nose:    selected_symptoms.append("runny_nose")
        if sel_headache:      selected_symptoms.append("headache")
        if sel_body_ache:     selected_symptoms.append("body_ache")
        if sel_fatigue:       selected_symptoms.append("fatigue")
        if sel_nausea:        selected_symptoms.append("nausea")
        if sel_vomiting:      selected_symptoms.append("vomiting")
        if sel_diarrhea:      selected_symptoms.append("diarrhea")
        if sel_abdom_pain:    selected_symptoms.append("abdominal_pain")
        if sel_rash:          selected_symptoms.append("rash")
        if sel_joint_pain:    selected_symptoms.append("joint_pain")
        if sel_chills:        selected_symptoms.append("chills")
        if sel_loss_ts:       selected_symptoms.append("loss_of_taste_smell")
        if sel_sob:           selected_symptoms.append("shortness_of_breath")
        if sel_sweating:      selected_symptoms.append("sweating")
        if sel_light_sens:    selected_symptoms.append("light_sensitivity")
        if sel_loss_app:      selected_symptoms.append("loss_of_appetite")

        # Show a live count of how many symptoms have been selected
        st.markdown(
            f'<p style="font-size:0.82rem;color:#6b7a8d;">'
            f"✅ {len(selected_symptoms)} symptom(s) selected</p>",
            unsafe_allow_html=True,
        )

        # Primary action button: triggers prediction pipeline
        analyse_clicked = st.button("🔍 Analyse Symptoms", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)  # Close context card

    # ─────────────────────────────────────────────────────────────────────
    # RESULTS COLUMN
    # ─────────────────────────────────────────────────────────────────────
    with col_results:
        # Show a placeholder until the user clicks "Analyse Symptoms"
        if not analyse_clicked:
            st.markdown(
                """
                <div class="card" style="text-align:center;padding:3rem 2rem;">
                    <div style="font-size:3rem;">🩺</div>
                    <h3 style="margin-top:1rem;">Ready to Analyse</h3>
                    <p style="color:#6b7a8d;font-size:0.88rem;">
                        Select your symptoms on the left and click
                        <strong>Analyse Symptoms</strong> to see results.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # ── Safety layer checks ───────────────────────────────────────────
            # high_fever_days = illness_days only counts if high fever is present
            hf_days = illness_days if "high_fever" in selected_symptoms else 0
            warnings_list = safety_check(selected_symptoms, hf_days)

            # Render warning/error banners for any triggered safety rules
            for w in warnings_list:
                is_urgent = "🚨" in w  # Distinguish urgent (red) from advisory (amber)
                banner_class = "urgent-banner" if is_urgent else "risk-banner"
                st.markdown(
                    f'<div class="{banner_class}">{w}</div>',
                    unsafe_allow_html=True,
                )

            # Only show predictions if at least one symptom was selected
            if len(selected_symptoms) > 0:
                # Build the binary feature vector from selected symptoms
                X_input = build_symptom_vector(selected_symptoms)

                # Get class-probability estimates from the Random Forest
                # predict_proba returns an array of shape (1, n_classes)
                proba = rf_model.predict_proba(X_input)[0]

                # Map each class name to its probability
                class_probs = dict(zip(rf_model.classes_, proba))

                # Sort conditions by descending probability and keep top 3
                top3 = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:3]

                # ── Results header ────────────────────────────────────────────
                st.markdown("### 📋 Top 3 Likely Conditions")
                st.markdown(
                    '<p style="font-size:0.82rem;color:#6b7a8d;margin-top:-0.6rem;">'
                    "Based on reported symptoms. Not a diagnosis.</p>",
                    unsafe_allow_html=True,
                )

                # Render one result card per top condition
                for rank, (condition, prob) in enumerate(top3):
                    # Highlight the top result with a teal border
                    card_class = "result-card top-result" if rank == 0 else "result-card"
                    bc = badge_class(prob)     # CSS class for the confidence badge
                    bl = badge_label(prob)     # Human-readable label
                    pct = round(prob * 100, 1) # Probability as a percentage

                    info = CONDITION_INFO.get(condition, {})
                    description = info.get("description", "")
                    self_care   = info.get("self_care", "")
                    see_doctor  = info.get("see_doctor", "")

                    # Render the condition result card with badge and description
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                                <h4>{'🥇 ' if rank==0 else ''}{condition}</h4>
                                <span class="badge {bc}">{bl} · {pct}%</span>
                            </div>
                            <p style="font-size:0.82rem;color:#374151;margin:0 0 0.5rem 0;">{description}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Native Streamlit progress bar for visual probability representation
                    st.progress(prob)

                    # Expandable self-care and when-to-see-a-doctor advice
                    with st.expander(f"💡 Self-care & advice for {condition}"):
                        st.markdown(f"**🏠 Self-care:** {self_care}")
                        st.markdown(f"**🏥 When to see a doctor:** {see_doctor}")
                        # Reiterate that this is not a diagnosis inside every advice block
                        st.info(
                            "⚠️ This is NOT a diagnosis. Please consult a "
                            "qualified medical professional for proper evaluation."
                        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 📈 Model Performance")
    st.markdown(
        '<p style="color:#6b7a8d;font-size:0.88rem;margin-top:-0.8rem;">'
        "Training and evaluation results on 1 200 synthetic patients.</p>",
        unsafe_allow_html=True,
    )

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏆 Model Comparison")
    st.markdown(
        "Three classifiers are trained and compared. "
        "**Random Forest** is selected as the final model because it consistently "
        "achieves the highest accuracy and is robust to overfitting via ensemble averaging.",
    )
    # Display the comparison DataFrame with full width
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Random Forest performance metrics as styled stat cards ────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌲 Random Forest — Final Model Metrics")

    # Four metric cards in a single row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{rf_metrics["accuracy"]:.1%}</div>'
            f'<div class="metric-label">Accuracy</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{rf_metrics["precision"]:.1%}</div>'
            f'<div class="metric-label">Precision (weighted)</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{rf_metrics["recall"]:.1%}</div>'
            f'<div class="metric-label">Recall (weighted)</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{rf_metrics["f1"]:.1%}</div>'
            f'<div class="metric-label">F1 Score (weighted)</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature importance bar chart ──────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔑 Top 10 Feature Importances")
    st.markdown(
        "Feature importance measures how much each symptom reduces prediction error "
        "across all 200 trees. Higher = more informative symptom."
    )
    # Display as a horizontal bar chart using Streamlit's native bar_chart
    chart_data = feat_imp_df.set_index("Symptom")["Importance"]
    st.bar_chart(chart_data, color="#1a7a8a")
    # Also show as a table for exact values
    st.dataframe(feat_imp_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Confusion Matrix (Test Set)")
    st.markdown(
        "Rows = true condition, Columns = predicted condition. "
        "Diagonal cells (correct predictions) should dominate."
    )
    st.dataframe(conf_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – DATA INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📊 Data Insights")
    st.markdown(
        '<p style="color:#6b7a8d;font-size:0.88rem;margin-top:-0.8rem;">'
        "Exploratory analysis of the synthetic dataset.</p>",
        unsafe_allow_html=True,
    )

    # ── Class balance chart ───────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚖️ Class Balance")
    st.markdown(
        "Each condition has exactly 150 patients before noise injection, "
        "giving a balanced dataset which prevents the model from being biased "
        "towards any single condition."
    )
    # Count rows per condition and display as a bar chart
    class_counts = df["condition"].value_counts().sort_index()
    st.bar_chart(class_counts, color="#0f4c81")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Symptom frequency per condition table ─────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Symptom Frequency per Condition")
    st.markdown(
        "Shows the proportion of patients with each symptom = 1, grouped by condition. "
        "Brighter values indicate higher prevalence."
    )

    # Compute mean of each binary symptom column, grouped by condition
    # .T transposes so conditions are columns and symptoms are rows
    freq_table = df.groupby("condition")[SYMPTOMS].mean().round(2)

    # Style the table with a gradient background for readability
    styled = freq_table.style.background_gradient(cmap="Blues", axis=None)
    st.dataframe(styled, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Dataset sample ────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗃️ Dataset Sample (first 20 rows)")
    # Show the first 20 rows so the user can inspect the raw data structure
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.markdown(
        f'<p style="font-size:0.8rem;color:#6b7a8d;">'
        f"Total rows: {len(df)} &nbsp;|&nbsp; "
        f"Features: {len(SYMPTOMS)} symptoms &nbsp;|&nbsp; "
        f"Classes: {df['condition'].nunique()}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – ABOUT PROJECT
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## ℹ️ About This Project")

    # About card
    st.markdown(
        """
        <div class="card">
        <h3>🎯 Project Goal</h3>
        <p>
            This application demonstrates how machine learning can support early
            disease screening in resource-limited settings, directly supporting
            <strong>UN Sustainable Development Goal 3</strong>: Good Health and
            Well-being. It is built as an educational tool to illustrate the full
            ML lifecycle — data generation, exploration, training, evaluation and
            real-time inference — using only open-source Python libraries.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Algorithm explanations
    st.markdown(
        """
        <div class="card">
        <h3>🤖 Algorithm Plain-English Guide</h3>

        <p><strong>🌲 Random Forest</strong><br>
        Imagine asking 200 doctors (each trained on a slightly different subset of
        patients) to vote on a diagnosis. The majority vote wins. Because each doctor
        sees different data, their errors are uncorrelated and tend to cancel out —
        this is <em>ensemble averaging</em>. Random Forest also randomly selects a
        subset of symptoms at each decision node, further decorrelating trees.</p>

        <p><strong>🌿 Decision Tree</strong><br>
        A single doctor that asks a series of yes/no questions ("Does the patient have
        high fever? → Yes → Do they have chills?…") until they reach a diagnosis. Easy
        to interpret, but a single tree can memorise training data (overfit).</p>

        <p><strong>📐 Logistic Regression</strong><br>
        Draws a straight boundary in symptom space and converts the signed distance to
        that boundary into a probability using the sigmoid function. Fast and
        interpretable, but struggles when illnesses are not linearly separable.</p>

        <p><strong>✂️ Train/Test Split</strong><br>
        We reserve 20 % of patients the model has never seen during training. If accuracy
        on this held-out set is similar to training accuracy, the model generalises well.</p>

        <p><strong>🔄 Cross-Validation (5-fold)</strong><br>
        The training set is cut into 5 equal slices. We train on 4 slices and test on the
        remaining 1, repeat this 5 times (rotating which slice is held out), then average
        the 5 scores. This gives a lower-variance estimate of generalisation performance
        than a single split.</p>

        <p><strong>📏 Accuracy, Precision, Recall, F1</strong><br>
        <em>Accuracy</em> = fraction of all predictions that are correct.<br>
        <em>Precision</em> = of all the times we predicted "Dengue", how often were we right?<br>
        <em>Recall</em> = of all actual Dengue cases, how many did we catch?<br>
        <em>F1</em> = 2 × (Precision × Recall) / (Precision + Recall) — a single number
        that balances both, especially useful when class sizes differ.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tech stack
    st.markdown(
        """
        <div class="card">
        <h3>🛠️ Technology Stack</h3>
        <ul>
            <li><strong>Streamlit</strong> — interactive web UI</li>
            <li><strong>scikit-learn</strong> — machine learning (Random Forest, Decision Tree, Logistic Regression)</li>
            <li><strong>NumPy</strong> — numerical array operations and random sampling</li>
            <li><strong>Pandas</strong> — tabular data manipulation and analysis</li>
        </ul>
        <p style="font-size:0.85rem;color:#6b7a8d;">
            No external files, no API keys, no CSV downloads required.
            Run with: <code>streamlit run app.py</code>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SDG alignment
    st.markdown(
        """
        <div class="card">
        <h3>🌍 SDG 3 Alignment</h3>
        <p>
            SDG 3 aims to <em>"ensure healthy lives and promote well-being for all at all ages."</em>
            AI-assisted screening tools can improve early detection in communities with limited
            access to specialists, reduce diagnostic delay, and help individuals make informed
            decisions about seeking professional care — all while keeping the human doctor firmly
            in the loop as the final decision-maker.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – PERMANENT FOOTER DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════

# Always-visible footer at the bottom of every page
st.markdown(
    """
    <div class="footer-disclaimer">
        🩺 <strong>AI Symptom Checker</strong> — Educational tool only.
        Not a medical diagnosis. &nbsp;|&nbsp;
        Dataset is synthetic and illustrative. &nbsp;|&nbsp;
        Always consult a qualified healthcare professional. &nbsp;|&nbsp;
        Aligned with <strong>UN SDG 3</strong>: Good Health &amp; Well-being.
    </div>
    """,
    unsafe_allow_html=True,
)
