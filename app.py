
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================================
# CREDITWISE AI
# CodeAlpha Machine Learning Internship
# Educational Credit Risk Prediction Application
# ==========================================================

st.set_page_config(
    page_title="CreditWise AI",
    page_icon="💳",
    layout="wide"
)

# --------------------------
# APPLICATION HEADER
# --------------------------

st.title("💳 CreditWise AI")
st.subheader("Explainable Credit Risk Prediction System")

st.write(
    "An educational machine learning application "
    "developed for the CodeAlpha Machine Learning Internship."
)

st.info(
    "This demonstration uses historical German credit data "
    "from 1973–1975. It must not be used for actual "
    "loan approvals or financial decisions."
)

# --------------------------
# LOAD TRAINED MODEL
# --------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "creditwise_model.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    artifact = load_model()
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_names = artifact["feature_names"]

except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()

st.success("Trained machine learning model loaded successfully!")

# --------------------------
# MODEL INFORMATION
# --------------------------

st.header("Model Information")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Algorithm", "Logistic Regression")

with c2:
    st.metric("Decision Threshold", f"{threshold:.2f}")

with c3:
    st.metric("Input Features", len(feature_names))

st.divider()

# --------------------------
# VERIFIED CATEGORY LABELS
# Based on South German Credit codebook
# --------------------------

CATEGORY_LABELS = {

    "laufkont": {
        1: "No checking account",
        2: "Negative checking balance",
        3: "Balance from 0 to below 200 DM",
        4: "Balance 200+ DM / salary for at least 1 year"
    },

    "moral": {
        0: "Past repayment delays",
        1: "Critical account / other credits elsewhere",
        2: "No previous credit / all repaid duly",
        3: "Existing credits repaid duly so far",
        4: "All credits at this bank repaid duly"
    },

    "verw": {
        0: "Other purpose",
        1: "New car",
        2: "Used car",
        3: "Furniture or equipment",
        4: "Radio or television",
        5: "Domestic appliances",
        6: "Repairs",
        7: "Education",
        8: "Vacation",
        9: "Retraining",
        10: "Business"
    },

    "sparkont": {
        1: "Unknown / no savings account",
        2: "Savings below 100 DM",
        3: "Savings from 100 to below 500 DM",
        4: "Savings from 500 to below 1,000 DM",
        5: "Savings of 1,000 DM or more"
    },

    "famges": {
        1: "Male, divorced or separated",
        2: "Female non-single / male single",
        3: "Male, married or widowed",
        4: "Female, single"
    },

    "buerge": {
        1: "No co-applicant or guarantor",
        2: "Co-applicant",
        3: "Guarantor"
    },

    "weitkred": {
        1: "Other installment plan with bank",
        2: "Other installment plan with store",
        3: "No other installment plan"
    },

    "wohn": {
        1: "Housing provided free",
        2: "Rented housing",
        3: "Own housing"
    },

    "telef": {
        1: "No registered landline",
        2: "Landline registered in applicant's name"
    },

    "gastarb": {
        1: "Foreign worker",
        2: "Not a foreign worker"
    },

    "beszeit": {
        1: "Unemployed",
        2: "Employed less than 1 year",
        3: "Employed 1 to below 4 years",
        4: "Employed 4 to below 7 years",
        5: "Employed 7 years or longer"
    },

    "rate": {
        1: "Installments: 35% or more of disposable income",
        2: "Installments: 25% to below 35%",
        3: "Installments: 20% to below 25%",
        4: "Installments: below 20%"
    },

    "wohnzeit": {
        1: "Residence less than 1 year",
        2: "Residence 1 to below 4 years",
        3: "Residence 4 to below 7 years",
        4: "Residence 7 years or longer"
    },

    "verm": {
        1: "Unknown / no property",
        2: "Car or other relevant property",
        3: "Building savings agreement / life insurance",
        4: "Real estate"
    },

    "bishkred": {
        1: "1 credit",
        2: "2–3 credits",
        3: "4–5 credits",
        4: "6 or more credits"
    },

    "beruf": {
        1: "Unemployed / unskilled non-resident",
        2: "Unskilled resident",
        3: "Skilled employee or official",
        4: "Manager / self-employed / highly qualified"
    },

    "pers": {
        1: "3 or more financial dependents",
        2: "0–2 financial dependents"
    }
}

FEATURE_LABELS = {
    "laufkont": "Checking Account Status",
    "moral": "Credit History",
    "verw": "Loan Purpose",
    "sparkont": "Savings Account",
    "famges": "Historical Personal Status",
    "buerge": "Other Debtors / Guarantor",
    "weitkred": "Other Installment Plans",
    "wohn": "Housing",
    "telef": "Registered Telephone",
    "gastarb": "Historical Foreign Worker Status",
    "beszeit": "Employment Duration",
    "rate": "Installment Rate",
    "wohnzeit": "Residence Duration",
    "verm": "Property",
    "bishkred": "Number of Existing Credits",
    "beruf": "Job Category",
    "pers": "Financial Dependents"
}

# --------------------------
# READ MODEL INPUT SCHEMA
# --------------------------

preprocessor = model.named_steps["preprocessor"]

encoder = (
    preprocessor.named_transformers_["cat"]
    .named_steps["encoder"]
)

categorical_columns = list(
    preprocessor.transformers_[1][2]
)

CATEGORY_OPTIONS = {
    name: [int(v) for v in categories]
    for name, categories in zip(
        categorical_columns,
        encoder.categories_
    )
}

NUMERICAL_FEATURES = ["laufzeit", "hoehe", "alter"]

# Verify that descriptions cover all encoded categories
for feature, options in CATEGORY_OPTIONS.items():
    if feature not in CATEGORY_LABELS:
        st.error(f"Missing feature description: {feature}")
        st.stop()

    missing = set(options) - set(CATEGORY_LABELS[feature])

    if missing:
        st.error(
            f"Missing category descriptions for {feature}: "
            f"{sorted(missing)}"
        )
        st.stop()

# Verify complete input schema
if set(feature_names) != (
    set(NUMERICAL_FEATURES) | set(CATEGORY_OPTIONS)
):
    st.error("The form does not match the saved model schema.")
    st.stop()

# --------------------------
# PREDICTION FORM
# --------------------------

st.header("Credit Risk Prediction")

st.write(
    "Enter a hypothetical historical applicant profile "
    "to explore the model's classification."
)

with st.form("credit_prediction_form"):

    st.subheader("Loan and Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.number_input(
            "Loan Duration (Months)",
            min_value=1,
            max_value=120,
            value=24,
            step=1
        )

    with col2:
        amount = st.number_input(
            "Historical Credit Amount (Transformed DM Units)",
            min_value=1,
            max_value=100000,
            value=3000,
            step=100,
            help="Historical credit amount. Not a modern currency value."
        )

    with col3:
        age = st.number_input(
            "Applicant Age (Years)",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

    inputs = {
        "laufzeit": int(duration),
        "hoehe": int(amount),
        "alter": int(age)
    }

    st.subheader("Financial and Historical Attributes")

    st.caption(
        "Descriptions are taken from the dataset codebook. "
        "Some fields are outdated or sensitive and are "
        "included only to reproduce the historical model."
    )

    columns = st.columns(3)

    for index, (feature, options) in enumerate(
        CATEGORY_OPTIONS.items()
    ):
        with columns[index % 3]:

            selected_code = st.selectbox(
                FEATURE_LABELS.get(feature, feature),
                options=options,
                format_func=lambda code, f=feature:
                    CATEGORY_LABELS[f][code],
                key=f"input_{feature}",
                help=f"Historical dataset variable: {feature}"
            )

            inputs[feature] = int(selected_code)

    submitted = st.form_submit_button(
        "Predict Credit Risk",
        type="primary",
        use_container_width=True
    )

# --------------------------
# GENERATE MODEL PREDICTION
# --------------------------

if submitted:

    input_df = pd.DataFrame(
        [inputs],
        columns=feature_names
    )

    if input_df.isnull().any().any():
        st.error("Incomplete input data.")
        st.stop()

    try:
        model_classes = list(model.classes_)
        bad_class_index = model_classes.index(1)

        risk_score = float(
            model.predict_proba(input_df)[0, bad_class_index]
        )

    except Exception as error:
        st.error("Prediction could not be generated.")
        st.exception(error)
        st.stop()

    prediction = int(risk_score >= threshold)

    st.divider()
    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model-Estimated Bad-Credit Score",
            f"{risk_score:.2%}"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

    st.progress(
        min(max(risk_score, 0.0), 1.0)
    )

    if prediction == 1:
        st.warning(
            "Model Classification: Bad Credit Risk"
        )
    else:
        st.success(
            "Model Classification: Good Credit Risk"
        )

    st.write(
        "This classification is based on a fixed threshold "
        "selected during model development."
    )

    st.info(
        "The model score is not a validated probability of "
        "real-world default. This tool is an educational "
        "demonstration and cannot assess actual applicants "
        "or make lending decisions."
    )

# --------------------------
# MODEL LIMITATIONS
# --------------------------

with st.expander("About the model and its limitations"):

    st.write(
        "Dataset: UCI South German Credit (1973–1975)."
    )

    st.write(
        "Independent test accuracy: 68.50%. "
        "Bad-credit recall: 61.67%. "
        "F1-score: 0.5401. ROC-AUC: 0.7418."
    )

    st.write(
        "The dataset contains historical and sensitive "
        "attributes, including combined personal-status "
        "information and foreign-worker status. "
        "The model has not been validated for modern "
        "lending, fairness, or probability calibration."
    )

    st.write(
        "These performance values were obtained from "
        "the project's held-out evaluation."
    )

st.divider()

st.caption(
    "CreditWise AI | CodeAlpha Machine Learning Internship "
    "| Dataset: UCI South German Credit"
)
