# 🩺 SwasthAI — AI-Powered Early Disease Screening Assistant

> A machine-learning web app that screens for 8 common illnesses from 20 binary symptoms — built for education, aligned with **UN SDG 3: Good Health and Well-being**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://[paste-your-streamlit-app-link])

---

## 🌐 Live Demo

**[👉 Try the app here]([https://[paste-your-streamlit-app-link](https://swasthai-l8sodya4cuhqzplycfdp6c.streamlit.app/)])**

No installation needed — runs entirely in the browser.

---

## 📋 Overview

### The Problem

In Tier-2, Tier-3 and rural areas, access to qualified doctors is limited. Patients often delay seeking care because they cannot easily identify whether their symptoms require urgent attention. Late detection of common but serious illnesses — such as Dengue, Malaria or Typhoid — increases risk of complications.

### The Solution

SwasthAI is a Streamlit web application that lets a user select their current symptoms, enter their age and duration of illness, and receive a ranked list of the three most likely conditions with probability scores and plain-language self-care advice. It trains three machine learning classifiers on a synthetic dataset generated entirely inside the code — no external files, no API keys, no CSV downloads required. The app is designed as an educational screening aid, not a replacement for a doctor.

---

## ✨ Key Features

- **20 binary symptom checkboxes** grouped into four categories — General, Respiratory, Digestive and Other — arranged in a compact 3-column layout.
- **Patient context inputs** — age (1–100) and days of illness (1–30) via sliders.
- **Top-3 condition predictions** with probability percentages, confidence badges and progress bars, each with a one-line description and expandable self-care advice.
- **Three-model comparison** — Random Forest (200 trees), Decision Tree (depth 8) and Logistic Regression — with test accuracy and 5-fold cross-validation mean accuracy displayed in a table.
- **Random Forest final model metrics** — accuracy, weighted precision, weighted recall and weighted F1 shown as styled stat cards.
- **Top-10 feature importances** visualised as a bar chart, showing which symptoms most influence predictions.
- **Confusion matrix** as a labelled DataFrame on the Model Performance tab.
- **Symptom frequency heatmap** — mean symptom prevalence per condition, styled with a blue gradient.
- **Safety warning layer** — three automatic rules fire before results are shown:
  - No symptoms selected → advisory banner.
  - `shortness_of_breath` selected → urgent red alert advising emergency care.
  - `high_fever` selected for 5 or more days → amber warning to consult a doctor promptly.
- **Custom SaaS-style UI** — gradient hero banner, white rounded cards, teal accent colour, Inter font, hidden Streamlit chrome, and responsive column layout — all injected via a single CSS block.
- **Four navigation tabs** — Symptom Checker, Model Performance, Data Insights, About Project.
- **Permanent footer disclaimer** on every tab.

---

## 🌍 SDG 3 Relevance

**UN Sustainable Development Goal 3** aims to *"ensure healthy lives and promote well-being for all at all ages."* AI-assisted screening tools can reduce diagnostic delay in communities with limited specialist access by helping individuals recognise symptom patterns that warrant clinical attention. SwasthAI demonstrates how open-source machine learning — running on a single Python file with no paid services — can support early awareness and informed decision-making. The app always keeps the human doctor as the final decision-maker and never presents results as a clinical diagnosis.

---

## 🛠️ Tech Stack

| Library | Version | Role in this project |
|---|---|---|
| `streamlit` | 1.x | Web UI framework — layout, widgets, tabs, caching |
| `scikit-learn` | 1.x | RandomForestClassifier, DecisionTreeClassifier, LogisticRegression, train_test_split, cross_val_score, metrics |
| `numpy` | 1.x | Synthetic data generation (`np.random.binomial`, `np.random.seed`), array operations |
| `pandas` | 1.x | DataFrames for the dataset, confusion matrix, feature importance table, symptom frequency table |

No other libraries are used. The standard library `warnings` module is imported to suppress harmless convergence warnings from LogisticRegression.

---

## ⚙️ How It Works — ML Workflow

### 1. Synthetic Data Generation

A dictionary (`CONDITION_SYMPTOM_PROBS`) defines the probability of each of the 20 symptoms being present for each of the 8 conditions. For example, Dengue has `high_fever: 0.9`, `joint_pain: 0.8`, `rash: 0.5`; Migraine has `headache: 0.95`, `light_sensitivity: 0.9` and no fever.

For each condition, 150 patient rows are generated by drawing each symptom independently from a Bernoulli distribution using `np.random.binomial(1, p)`. Symptoms not listed for a condition use a background probability of 0.05. The random seed is fixed at `np.random.seed(42)` for reproducibility. After all 1 200 rows are assembled, 5% of labels are randomly reassigned to a different condition to add noise and prevent perfect class separation.

### 2. Train / Test Split

The 1 200-row dataset is split 80/20 using `train_test_split` with `stratify=y` and `random_state=42`. Stratification ensures every condition is proportionally represented in both the training set (960 rows) and the test set (240 rows).

### 3. Model Training and Comparison

Three classifiers are trained on the training set:

| Model | Parameters |
|---|---|
| `RandomForestClassifier` | `n_estimators=200`, `random_state=42` |
| `DecisionTreeClassifier` | `max_depth=8`, `random_state=42` |
| `LogisticRegression` | `max_iter=1000`, `random_state=42` |

Test-set accuracy and 5-fold cross-validation mean accuracy are computed for each and displayed in a comparison table.

### 4. Final Model — Random Forest

Random Forest is selected as the final prediction model because ensemble averaging over 200 trees reduces overfitting compared to a single Decision Tree and handles the non-linear structure of the symptom space better than Logistic Regression.

Evaluation metrics on the held-out test set are computed using weighted averages across all 8 classes: accuracy, precision, recall and F1. The confusion matrix (8×8) and top-10 feature importances (mean decrease in impurity) are also computed and displayed.

### 5. Prediction

When the user clicks **Analyse Symptoms**, the selected symptom checkboxes are converted into a binary vector of length 20 using `build_symptom_vector`. The Random Forest's `predict_proba` method returns a probability for each of the 8 conditions; the top 3 are ranked by descending probability and displayed with confidence badges and progress bars.

---

## 📊 Model Performance

> *Run the app locally to see exact numbers. The values below are placeholders — fill them in after your first run.*

### Random Forest — Final Model (Test Set, Weighted Average)

| Metric | Score |
|---|---|
| Accuracy | 83.8% |
| Precision (weighted) | 84.8% |
| Recall (weighted) | 83.8% |
| F1 Score (weighted) | 83.9% |

### Three-Model Comparison

| Model | Test Accuracy | 5-Fold CV Mean Accuracy |
|---|---|---|
| Random Forest (200 trees) | 0.838 | 0.864 |
| Decision Tree (depth 8) | 0.767 | 0.727 |
| Logistic Regression | 0.850 | 0.881 |

---

## 🗂️ Project Structure

```
swasthai/
├── app.py                 # Complete self-contained Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── screenshots/
    ├── checker.png
    ├── results.png
    └── performance.png
```

---

## 🚀 Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/[your-username]/swasthai.git
   cd swasthai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the app**
   ```bash
   streamlit run app.py
   ```

4. Open the URL printed in the terminal (usually `http://localhost:8501`) in your browser.

### `requirements.txt`

```
streamlit
pandas
numpy
scikit-learn
```

---

## ⚠️ Limitations

- **Synthetic dataset only.** All 1 200 patient records are generated programmatically using manually set probability values. The data has not been collected from real patients and has not undergone any clinical validation.
- **Not clinically validated.** The symptom probabilities are illustrative approximations, not derived from epidemiological studies or medical literature.
- **Limited scope.** The app screens for 8 conditions only: Common Cold, Influenza, Dengue, Malaria, Typhoid, Gastroenteritis, COVID-19-like Illness and Migraine. Many other illnesses share overlapping symptoms and are not modelled.
- **Not a substitute for a doctor.** No machine learning model, however accurate, can replace a clinical examination, laboratory tests or professional medical judgement.

---

## 🔭 Future Scope

- Replace the synthetic dataset with real, anonymised clinical data to improve reliability.
- Expand coverage to more conditions (e.g. pneumonia, tuberculosis, chickenpox).
- Add multilingual support for regional Indian languages to serve a wider rural population.
- Integrate a doctor-referral or telemedicine booking feature.
- Build a mobile-friendly Progressive Web App (PWA) or a React Native front-end.

---

## 🔴 Disclaimer

> **This application is an educational project created for the IBM SkillsBuild Machine Learning & AI Internship.**
>
> It is **NOT a medical diagnosis tool**. The dataset is synthetic and illustrative. Predictions are based on a machine learning model trained on artificially generated data and carry no clinical validity. **Always consult a qualified and licensed healthcare professional for any medical concern.** Do not use this tool to make health decisions.

---

## 👤 Author & Acknowledgements

**Kapil Singh Songare**
University institute of technology, RGPV Bhopal, M.P.
IBM SkillsBuild Machine Learning & AI Internship
Edunet Foundation / AICTE

**Acknowledgements**

- [IBM SkillsBuild](https://skillsbuild.org/) — for the internship programme and learning resources.
- [Edunet Foundation](https://edunetfoundation.org/) & [AICTE](https://www.aicte-india.org/) — for organising and supporting the internship.
- [Streamlit](https://streamlit.io/) — for the open-source web-app framework.
- [scikit-learn](https://scikit-learn.org/) — for the machine learning library.
- United Nations — for the [Sustainable Development Goals](https://sdgs.un.org/goals) framework.

---

*Built with Python · Streamlit · scikit-learn · UN SDG 3*
