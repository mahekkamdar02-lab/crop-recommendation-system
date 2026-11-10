"""Location-aware, explainable crop recommender (Streamlit).

Run:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from location import MONTHS, get_features
from model_utils import explain, global_importance, load_bundle, top_k

try:
    import folium
    from streamlit_folium import st_folium
    HAS_MAP = True
except ImportError:
    HAS_MAP = False

MODEL_PATH = "models/crop_model.joblib"
st.set_page_config(page_title="Location-Aware Crop Recommender", page_icon="🌱", layout="wide")

DEFAULTS = {
    "lat": 20.59, "lon": 78.96, "season": ["JUN", "JUL", "AUG", "SEP"],
    "f_N": 90.0, "f_P": 42.0, "f_K": 43.0, "f_temperature": 25.0,
    "f_humidity": 80.0, "f_ph": 6.5, "f_rainfall": 200.0,
    "msgs": [], "fetched": [], "last_click": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


def fetch_location_data():
    """Button callback: fill climate + soil pH from the selected coordinates."""
    if not st.session_state.season:
        st.session_state.msgs = ["Pick at least one growing-season month."]
        return
    feats, warns = get_features(st.session_state.lat, st.session_state.lon, st.session_state.season)
    for k, v in feats.items():
        st.session_state[f"f_{k}"] = float(v)
    st.session_state.fetched = sorted(feats)
    st.session_state.msgs = warns


try:
    bundle = get_bundle()
except FileNotFoundError:
    st.error("No trained model found. Run `python train.py --data data/Crop_recommendation.csv` first.")
    st.stop()

st.title("🌱 Location-Aware Crop Recommender")
st.caption("Pick a location → fetch local climate and soil pH → add N, P, K → get the top crops with an explanation.")

col_map, col_in = st.columns([3, 2])

with col_map:
    if HAS_MAP:
        m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=5)
        folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)
        out = st_folium(m, height=380, use_container_width=True, key="map", returned_objects=["last_clicked"])
        click = out.get("last_clicked") if out else None
        if click:
            c = (round(click["lat"], 5), round(click["lng"], 5))
            if c != st.session_state.last_click:  # act on each click only once
                st.session_state.last_click = c
                st.session_state.lat, st.session_state.lon = c
                st.rerun()
    else:
        st.info("Install `folium` and `streamlit-folium` for the map picker. You can still type coordinates.")

with col_in:
    st.subheader("1. Location")
    st.number_input("Latitude", -90.0, 90.0, key="lat", step=0.01, format="%.4f")
    st.number_input("Longitude", -180.0, 180.0, key="lon", step=0.01, format="%.4f")
    st.multiselect("Growing season (months)", MONTHS, key="season")
    st.button("Fetch climate & soil for this location", on_click=fetch_location_data, type="primary")
    for w in st.session_state.msgs:
        st.warning(w + " Enter the value manually below.")
    if st.session_state.fetched:
        st.success("Filled from location: " + ", ".join(st.session_state.fetched))

st.subheader("2. Soil and climate inputs (editable)")
c1, c2, c3, c4 = st.columns(4)
c1.number_input("Nitrogen N", 0.0, 300.0, key="f_N", step=1.0, help="Typically from a soil-health card")
c1.number_input("Phosphorus P", 0.0, 300.0, key="f_P", step=1.0)
c2.number_input("Potassium K", 0.0, 300.0, key="f_K", step=1.0)
c2.number_input("Soil pH", 3.0, 10.0, key="f_ph", step=0.1)
c3.number_input("Temperature (°C)", -10.0, 55.0, key="f_temperature", step=0.5)
c3.number_input("Humidity (%)", 0.0, 100.0, key="f_humidity", step=1.0)
c4.number_input("Rainfall (mm / month)", 0.0, 1500.0, key="f_rainfall", step=5.0)

row = pd.DataFrame([{f: st.session_state[f"f_{f}"] for f in bundle["features"]}])

if st.button("3. Recommend crops", type="primary"):
    # warn when inputs are outside what the model was trained on
    for f, (lo, hi) in bundle.get("ranges", {}).items():
        v = float(row[f].iloc[0])
        if v < lo or v > hi:
            st.warning(f"{f} = {v} is outside the training range ({lo:.1f}–{hi:.1f}); treat this result with caution.")

    results, idx = top_k(bundle, row, 3)
    st.subheader("Top recommendations")
    cols = st.columns(3)
    for i, (crop, p) in enumerate(results):
        cols[i].metric(f"#{i + 1}  {crop.title()}", f"{p:.0%}")
        cols[i].progress(min(max(p, 0.0), 1.0))
    if results[0][1] < 0.5:
        st.info("Confidence is low – the inputs fit several crops about equally well.")

    st.subheader(f"Why {results[0][0].title()}?")
    s = explain(bundle, row, int(idx[0]))
    if s is not None:
        s = s.reindex(s.abs().sort_values().index)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(s.index, s.values, color=["#2e7d32" if v >= 0 else "#c0392b" for v in s.values])
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("SHAP value (green pushes towards this crop, red pushes away)")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Install `shap` for per-prediction explanations. Showing global feature importance instead.")
        gi = global_importance(bundle)
        if gi is not None:
            st.bar_chart(gi.sort_values())


with st.expander("Model validation results"):
    st.write(
        f"Selected model: **{bundle['model_name']}** "
        f"({bundle['n_rows']} dataset rows)"
    )

    st.subheader("Final hold-out test results")

   
test_metrics = bundle.get("test_metrics", {})

if test_metrics:
    rows = []

    for validation_name, metrics in test_metrics.items():
        if isinstance(metrics, dict):
            for metric_name, score in metrics.items():
                if isinstance(score, (int, float)):
                    rows.append({
                        "Validation": validation_name,
                        "Metric": metric_name,
                        "Score": score
                    })
        elif isinstance(metrics, (int, float)):
            rows.append({
                "Validation": "Hold-out test",
                "Metric": validation_name,
                "Score": metrics
            })

    if rows:
        results_df = pd.DataFrame(rows)
        results_df["Score"] = results_df["Score"].map(
            lambda x: f"{x:.4f}"
        )
        st.dataframe(results_df, hide_index=True)
    else:
        st.warning("No numeric test metrics were found.")
else:
    st.info("Final test metrics are not available in this model file.")


    st.subheader("Training-set cross-validation")
    for name, results in bundle["metrics"].items():
        st.write(f"**{name}**")
        st.dataframe(pd.DataFrame(results).T)

    st.caption(
        "The test set was held out during model selection. "
        "Random splitting does not establish performance on unseen "
        "geographical regions or future seasons."
    )

with st.expander("Limitations"):
    st.markdown(
        "- The model is trained on a benchmark dataset without coordinates; location acts through the climate/soil lookup.\n"
        "- Climate is a long-term seasonal average, not a forecast.\n"
        "- Soil pH comes from a global 250 m product and may differ from your field; N, P, K must be entered by you.\n"
        "- Recommendations are decision support, not agronomic advice. Confirm with a local agriculture officer."
    )
