
import joblib
from pathlib import Path

# Load the saved model
model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "creditwise_model.joblib"
)

artifact = joblib.load(model_path)

pipeline = artifact["model"]
preprocessor = pipeline.named_steps["preprocessor"]

# Display required input columns
print("REQUIRED INPUT FEATURES:")
for feature in artifact["feature_names"]:
    print(feature)

# Inspect categorical values learned during training
encoder = preprocessor.named_transformers_[
    "cat"
].named_steps["encoder"]

categorical_columns = preprocessor.transformers_[1][2]

print("\nSUPPORTED CATEGORICAL VALUES:")

for name, categories in zip(
    categorical_columns,
    encoder.categories_
):
    print(f"{name}: {categories.tolist()}")

print("\nDecision threshold:", artifact["threshold"])


import pandas as pd

# Reconstruct the demonstration input from the app
# Numerical default values and first categorical options
inputs = {
    "laufzeit": 24,
    "hoehe": 3000,
    "alter": 30
}

for name, categories in zip(
    categorical_columns,
    encoder.categories_
):
    inputs[name] = int(categories[0])

# Arrange columns exactly as expected by the model
input_df = pd.DataFrame(
    [inputs],
    columns=artifact["feature_names"]
)

# Generate prediction directly
probability = pipeline.predict_proba(input_df)[0, 1]

prediction = int(
    probability >= artifact["threshold"]
)

print("\nINDEPENDENT APPLICATION CHECK")
print(f"Model score: {probability:.2%}")
print(
    "Classification:",
    "Bad Credit Risk" if prediction == 1
    else "Good Credit Risk"
)
print("Decision threshold:", artifact["threshold"])
