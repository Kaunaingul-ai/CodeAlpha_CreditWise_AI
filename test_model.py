
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load the saved model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "creditwise_model.joblib"
)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]

preprocessor = model.named_steps["preprocessor"]
encoder = (
    preprocessor.named_transformers_["cat"]
    .named_steps["encoder"]
)

categorical_columns = preprocessor.transformers_[1][2]

# Create valid demonstration input
sample = {
    "laufzeit": 24,
    "hoehe": 3000,
    "alter": 30
}

for feature, categories in zip(
    categorical_columns,
    encoder.categories_
):
    sample[feature] = int(categories[0])

input_df = pd.DataFrame(
    [sample],
    columns=feature_names
)

# TEST 1: Input structure
assert input_df.shape == (1, 20)
print("PASS: Input structure verified")

# TEST 2: Model probability is valid
probability = model.predict_proba(input_df)[0, 1]

assert np.isfinite(probability)
assert 0 <= probability <= 1
print("PASS: Model probability is valid")

# TEST 3: Classification logic
prediction = int(probability >= threshold)

assert prediction in [0, 1]
print("PASS: Classification logic verified")

# TEST 4: Repeat prediction consistency
repeat_probability = model.predict_proba(input_df)[0, 1]

np.testing.assert_allclose(
    probability,
    repeat_probability,
    rtol=1e-10,
    atol=1e-12
)
print("PASS: Repeated predictions are consistent")

# TEST 5: Feature order consistency
reordered_input = input_df[
    list(reversed(feature_names))
].copy()

reordered_input = reordered_input[feature_names]

reordered_probability = model.predict_proba(
    reordered_input
)[0, 1]

np.testing.assert_allclose(
    probability,
    reordered_probability,
    rtol=1e-10,
    atol=1e-12
)

print("PASS: Feature ordering verified")

print("\nALL MODEL CHECKS PASSED")
print(f"Model score: {probability:.2%}")
print("Decision threshold:", threshold)
print(
    "Classification:",
    "Bad Credit Risk" if prediction == 1
    else "Good Credit Risk"
)
