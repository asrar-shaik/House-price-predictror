import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model + preprocessing pipeline (cached across reruns)."""
    if not (os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE)):
        return None, None
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline


def run_inference(df: pd.DataFrame, model, pipeline) -> pd.DataFrame:
    """Apply the saved pipeline + model to a raw input dataframe."""
    result = df.copy()

    # Drop the target column if the user's file happens to include it
    if "median_house_value" in result.columns:
        result = result.drop(columns=["median_house_value"])

    transformed = pipeline.transform(result)
    predictions = model.predict(transformed)
    result["predicted_median_house_value"] = predictions
    return result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🏠House Price Predictor")
st.markdown(
    "A Random Forest regression model that predicts **median house value** "
    "for California housing districts"
)

model, pipeline = load_artifacts()

if model is None:
    st.error(
        "Model files (`model.pkl` / `pipeline.pkl`) were not found in the app "
        "directory. Run the training script once locally (it will create "
        "these files) and include them alongside `app.py` when you deploy."
    )
    st.stop()

with st.sidebar:
    st.header("About this app")
    st.markdown(
        """
        **Pipeline**
        - Median imputation + standard scaling for numeric features
        - One-hot encoding for `ocean_proximity`
        - `RandomForestRegressor` model

        **How to use**
        1. Upload a CSV with the same feature columns used in training
           (e.g. `longitude`, `latitude`, `housing_median_age`,
           `total_rooms`, `total_bedrooms`, `population`, `households`,
           `median_income`, `ocean_proximity`).
        2. Get predicted median house values instantly.
        3. Download the results as a CSV.
        """
    )
    st.divider()
    st.caption("Built by [Your Name] · scikit-learn + Streamlit")

st.subheader("1. Upload your data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

sample_toggle = st.toggle("I don't have a file — use a small built-in sample instead")

input_df = None

if sample_toggle:
    input_df = pd.DataFrame(
        {
            "longitude": [-122.23, -118.30, -121.98],
            "latitude": [37.88, 34.05, 37.35],
            "housing_median_age": [41, 25, 15],
            "total_rooms": [880, 3200, 1500],
            "total_bedrooms": [129, 700, 300],
            "population": [322, 1800, 900],
            "households": [126, 650, 320],
            "median_income": [8.3, 3.5, 5.1],
            "ocean_proximity": ["NEAR BAY", "<1H OCEAN", "INLAND"],
        }
    )
elif uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read that file as a CSV: {e}")

if input_df is not None:
    st.subheader("2. Preview")
    st.dataframe(input_df.head(20), use_container_width=True)

    st.subheader("3. Predictions")
    try:
        with st.spinner("Running the model..."):
            result_df = run_inference(input_df, model, pipeline)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows predicted", len(result_df))
        col2.metric(
            "Average predicted value",
            f"${result_df['predicted_median_house_value'].mean():,.0f}",
        )
        col3.metric(
            "Max predicted value",
            f"${result_df['predicted_median_house_value'].max():,.0f}",
        )

        st.dataframe(result_df, use_container_width=True)

        if {"latitude", "longitude"}.issubset(result_df.columns):
            st.subheader("Map of predicted values")
            map_df = result_df.rename(columns={"latitude": "lat", "longitude": "lon"})
            st.map(map_df[["lat", "lon"]])

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions as CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(
            "Something went wrong while running the model on this file. "
            "Make sure the columns match what the model was trained on.\n\n"
            f"Details: {e}"
        )
else:
    st.info("Upload a CSV file above (or toggle the sample data) to get started.")
