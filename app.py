from pathlib import Path
import joblib
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# AEROGUARD | Predictive Maintenance Dashboard
# Keep this file beside the "data" and "charts" folders.
# =========================================================


st.set_page_config(
    page_title="AeroGuard | Engine Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHART_DIR = BASE_DIR / "charts"
MODEL_PATH = DATA_DIR / "predictive_maintenance_model.pkl"

RAW_COLUMNS = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

MODEL_RESULTS = pd.DataFrame({
    "Model": [
        "Linear Regression", "Polynomial Regression", "Decision Tree",
        "Random Forest", "Gradient Boosting"
    ],
    "MAE": [25.16, 22.56, 35.66, 24.01, 22.70],
    "RMSE": [31.66, 29.03, 49.15, 31.70, 29.76],
    "R²": [0.768, 0.804, 0.439, 0.767, 0.794],
})

TEST_METRICS = {"MAE": 20.69, "MSE": 703.04, "RMSE": 26.51, "R²": 0.593}

# -------------------- LIGHT THEME --------------------
st.markdown("""
<style>
:root {
    --ink: #172b4d;
    --muted: #667892;
    --line: #e3eaf3;
    --blue: #2f6fed;
    --teal: #13a89e;
}
.stApp {
    background: linear-gradient(135deg, #f6f9ff 0%, #eef5fc 52%, #f8fbff 100%);
    color: var(--ink);
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e4ebf4;
}
[data-testid="stSidebar"] * { color: #203654; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #172b4d; letter-spacing: -0.4px; }
p, label, .stCaption { color: #526681; }
.hero {
    padding: 28px 32px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(115deg, #173b73 0%, #2f6fed 62%, #36b8c5 100%);
    box-shadow: 0 12px 30px rgba(47,111,237,.15);
    margin-bottom: 22px;
}
.hero-kicker {font-size: 12px; font-weight: 800; letter-spacing: 2.3px; opacity: .86;}
.hero h1 {color: white; font-size: 36px; margin: 7px 0 5px 0;}
.hero p {color: #edf5ff; margin: 0; font-size: 15px;}
.metric-card {
    background: rgba(255,255,255,.94);
    border: 1px solid #e3eaf3;
    border-radius: 17px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(34,65,110,.045);
    min-height: 112px;
}
.metric-label {font-size: 12px; color: #71819a; font-weight: 700; text-transform: uppercase; letter-spacing: .7px;}
.metric-value {font-size: 27px; color: #1d3760; font-weight: 800; margin-top: 8px;}
.metric-foot {font-size: 12px; color: #8492a7; margin-top: 2px;}
.panel {
    background: #ffffff;
    border: 1px solid #e4ebf4;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(34,65,110,.04);
}
div[data-testid="stMetric"] {
    background: white; border: 1px solid #e3eaf3;
    border-radius: 15px; padding: 15px 17px;
    box-shadow: 0 5px 18px rgba(34,65,110,.04);
}
div[data-testid="stMetricValue"] { color: #245ec7; }
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 0; font-weight: 700;
    background: #2f6fed; color: white; min-height: 42px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #1d55c1; color: white;
}
div[data-testid="stTabs"] button {font-weight: 650;}
hr {border-color: #e5ebf3;}
.small-note {color: #72829a; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# -------------------- HELPERS --------------------
@st.cache_resource
def load_model_bundle():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), None
        except Exception as exc:
            return None, str(exc)
    return None, "Model file not found."

@st.cache_data
def load_raw_data(path_string, modified_time):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)
    except Exception:
        return None

def get_train_data():
    path = DATA_DIR / "train_FD001.txt"
    if not path.exists():
        path = BASE_DIR / "train_FD001.txt"
    return load_raw_data(str(path), path.stat().st_mtime if path.exists() else 0)

def get_test_data():
    path = DATA_DIR / "test_FD001.txt"
    if not path.exists():
        path = BASE_DIR / "test_FD001.txt"
    return load_raw_data(str(path), path.stat().st_mtime if path.exists() else 0)

def make_rul_data(df):
    if df is None or df.empty:
        return None
    result = df.copy()
    last_cycle = result.groupby("engine_id")["cycle"].transform("max")
    result["RUL"] = last_cycle - result["cycle"]
    return result

def metric_card(label, value, foot=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True
    )

def style_plot(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Arial", color="#334968", size=12),
        margin=dict(l=12, r=12, t=45, b=12),
        title_font=dict(size=16, color="#203654"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="#edf1f7", zerolinecolor="#edf1f7"),
        yaxis=dict(showgrid=True, gridcolor="#edf1f7", zerolinecolor="#edf1f7"),
    )
    return fig

def image_files():
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    found = []
    seen_hashes = set()

    for folder in (CHART_DIR, DATA_DIR, BASE_DIR):
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in extensions:
                    file_hash = hashlib.sha256(p.read_bytes()).hexdigest()

                    if file_hash not in seen_hashes:
                        seen_hashes.add(file_hash)
                        found.append(p)

    return sorted(found, key=lambda p: p.name.lower())

# -------------------- DATA / MODEL --------------------
bundle, model_error = load_model_bundle()
train_raw = get_train_data()
test_raw = get_test_data()
train = make_rul_data(train_raw)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## ✈️ AeroGuard")
    st.caption("ENGINE HEALTH INTELLIGENCE")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Data Explorer", "Model Performance", "Visual Gallery",
         "Predict RUL", "Project Notes"],
        label_visibility="visible"
    )
    st.divider()
    st.markdown("**Project dataset**")
    st.caption("NASA C-MAPSS · FD001")
    st.markdown("**Selected model**")
    st.caption("Polynomial Regression + Ridge")
    st.markdown("**Status**")
    if bundle is not None:
        st.success("Model file loaded")
    else:
        st.warning("Model file not detected")
    st.caption("Tip: place saved charts in the `charts` folder.")

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">PREDICTIVE MAINTENANCE · MACHINE LEARNING</div>
  <h1>AeroGuard</h1>
  <p>Explore engine degradation, compare regression models, and estimate
  Remaining Useful Life from operating conditions and sensor measurements.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- OVERVIEW --------------------
if page == "Overview":
    st.subheader("Command centre")
    st.caption("A quick view of the dataset, selected model, and reported test performance.")

    engines = int(train_raw["engine_id"].nunique()) if train_raw is not None else 100
    train_rows = len(train_raw) if train_raw is not None else 20631
    test_rows = len(test_raw) if test_raw is not None else 13096

    cols = st.columns(4)
    with cols[0]: metric_card("Training records", f"{train_rows:,}", "NASA FD001 training file")
    with cols[1]: metric_card("Test records", f"{test_rows:,}", "NASA FD001 test file")
    with cols[2]: metric_card("Training engines", f"{engines:,}", "Distinct engine IDs")
    with cols[3]: metric_card("Model inputs", "25", "Cycle + settings + 21 sensors")

    st.markdown("### Final model · NASA test results")
    cols = st.columns(4)
    for col, (label, value, foot) in zip(cols, [
        ("MAE", "20.69 cycles", "Lower is better"),
        ("RMSE", "26.51 cycles", "Penalizes larger errors"),
        ("MSE", "703.04", "Squared-cycle error"),
        ("R² score", "0.593", "Test-set goodness of fit"),
    ]):
        with col:
            metric_card(label, value, foot)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Validation model comparison")
        metric_choice = st.selectbox("Choose a metric", ["MAE", "RMSE", "R²"], key="overview_metric")
        fig = px.bar(
            MODEL_RESULTS.sort_values(metric_choice, ascending=(metric_choice != "R²")),
            x="Model", y=metric_choice, color="Model",
            title=f"{metric_choice} across five models",
            hover_data={"MAE": ":.2f", "RMSE": ":.2f", "R²": ":.3f"}
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=metric_choice)
        st.plotly_chart(style_plot(fig, 360), use_container_width=True)
    with right:
        st.markdown("#### Project workflow")
        st.markdown("""
        <div class="panel">
        <b>01 · Load</b><br><span class="small-note">NASA C-MAPSS FD001 records</span><hr>
        <b>02 · Prepare</b><br><span class="small-note">Create engine-wise RUL targets</span><hr>
        <b>03 · Train</b><br><span class="small-note">Compare five regression algorithms</span><hr>
        <b>04 · Evaluate</b><br><span class="small-note">MAE, RMSE, MSE and R²</span><hr>
        <b>05 · Predict</b><br><span class="small-note">Estimate RUL from sensor readings</span>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Validation and test scores are different evaluations. The model-comparison "
        "page shows the previously reported validation metrics; the test cards show "
        "the final model's reported NASA test results."
    )

# -------------------- DATA EXPLORER --------------------
elif page == "Data Explorer":
    st.subheader("Data explorer")
    if train_raw is None:
        st.warning("Training data was not found. Put `train_FD001.txt` in the data folder.")
    else:
        st.caption("Use the filters to inspect engine records and sensor distributions.")
        min_engine = int(train_raw["engine_id"].min())
        max_engine = int(train_raw["engine_id"].max())
        engine_id = st.slider("Select engine ID", min_engine, max_engine, min_engine)
        engine_df = train[train["engine_id"] == engine_id].copy()

        a, b, c = st.columns(3)
        a.metric("Engine ID", engine_id)
        b.metric("Observed cycles", int(engine_df["cycle"].max()))
        c.metric("Starting RUL", int(engine_df["RUL"].max()))

        tab1, tab2, tab3 = st.tabs(["Engine timeline", "RUL distribution", "Dataset preview"])
        with tab1:
            fig = px.line(engine_df, x="cycle", y="RUL", markers=True,
                          title=f"RUL decreases over time · Engine {engine_id}")
            fig.update_traces(line=dict(color="#2f6fed", width=3))
            st.plotly_chart(style_plot(fig), use_container_width=True)
        with tab2:
            fig = px.histogram(train, x="RUL", nbins=40, title="Training RUL distribution",
                               color_discrete_sequence=["#13a89e"])
            st.plotly_chart(style_plot(fig), use_container_width=True)
        with tab3:
            st.dataframe(train_raw.head(100), use_container_width=True, hide_index=True)
            st.caption("Showing the first 100 rows of the raw training file.")
            st.download_button(
                "Download preview as CSV",
                data=train_raw.head(100).to_csv(index=False).encode("utf-8"),
                file_name="train_fd001_preview.csv",
                mime="text/csv"
            )

        st.markdown("#### Sensor distribution")
        sensor = st.selectbox("Choose a sensor", [f"sensor_{i}" for i in range(1, 22)])
        fig = px.histogram(train_raw, x=sensor, nbins=40, title=f"{sensor} values",
                           color_discrete_sequence=["#5b8def"])
        st.plotly_chart(style_plot(fig), use_container_width=True)

# -------------------- MODEL PERFORMANCE --------------------
elif page == "Model Performance":
    st.subheader("Model performance lab")
    st.caption("Interactive comparison using the project's reported validation scores.")
    metric = st.segmented_control("Metric", ["MAE", "RMSE", "R²"], default="MAE")
    metric = metric or "MAE"

    best_idx = MODEL_RESULTS[metric].idxmax() if metric == "R²" else MODEL_RESULTS[metric].idxmin()
    best = MODEL_RESULTS.loc[best_idx]

    a, b, c = st.columns(3)
    a.metric("Best model for selected metric", best["Model"])
    b.metric(f"Best {metric}", f"{best[metric]:.3f}" if metric == "R²" else f"{best[metric]:.2f}")
    c.metric("Models compared", len(MODEL_RESULTS))

    fig = px.bar(
        MODEL_RESULTS.sort_values(metric, ascending=(metric != "R²")),
        x="Model", y=metric, color="Model", text_auto=".3f",
        title=f"Model comparison · {metric}"
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=metric)
    st.plotly_chart(style_plot(fig, 400), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=MODEL_RESULTS["MAE"], y=MODEL_RESULTS["RMSE"],
        mode="markers+text", text=MODEL_RESULTS["Model"],
        textposition="top center", marker=dict(size=13, color="#2f6fed"),
        hovertemplate="<b>%{text}</b><br>MAE: %{x:.2f}<br>RMSE: %{y:.2f}<extra></extra>"
    ))
    fig.update_layout(title="Error trade-off: MAE vs RMSE", xaxis_title="MAE (cycles)",
                      yaxis_title="RMSE (cycles)")
    st.plotly_chart(style_plot(fig, 380), use_container_width=True)

    st.markdown("#### Validation score table")
    st.dataframe(
        MODEL_RESULTS.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}"}),
        use_container_width=True, hide_index=True
    )
    st.markdown("#### Final model test metrics")
    test_df = pd.DataFrame({"Metric": list(TEST_METRICS.keys()), "NASA test result": list(TEST_METRICS.values())})
    st.dataframe(test_df, use_container_width=True, hide_index=True)
    st.caption("Do not compare validation and test scores as if they came from the same data split.")

# -------------------- VISUAL GALLERY --------------------


elif page == "Visual Gallery":
    st.subheader("Visual Analytics Gallery")
    st.caption("All saved graphs from your project folders.")

    images = image_files()

    st.write(f"Images found: {len(images)}")

    if not images:
        st.warning("No images found. Check your charts folder.")
    else:
        cols_per_row = st.radio(
            "Graphs per row",
            [1, 2, 3],
            index=1,
            horizontal=True,
            key="gallery_layout_all"
        )

        for start in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)

            for col, image_path in zip(
                cols, images[start:start + cols_per_row]
            ):
                with col:
                    st.markdown(
                        f"**{image_path.stem.replace('_', ' ')}**"
                    )

                    try:
                        with open(image_path, "rb") as img_file:
                            image_bytes = img_file.read()

                        st.image(
                            image_bytes,
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(
                            f"Could not load {image_path.name}"
                        )
                        st.code(str(e))



# -------------------- PREDICTION --------------------
elif page == "Predict RUL":
    st.subheader("Estimate remaining useful life")
    st.caption("Enter operating settings and sensor measurements. Use real measurements where possible.")

    if bundle is None:
        st.error(f"Could not load the saved model. Expected path: {MODEL_PATH}")
        if model_error:
            st.caption(model_error)
    else:
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]

        # Defaults based on observed training data, when available.
        if train_raw is not None:
            medians = train_raw.median(numeric_only=True).to_dict()
        else:
            medians = {}

        with st.form("prediction_form"):
            st.markdown("#### Operating conditions")
            c1, c2, c3 = st.columns(3)
            with c1:
                cycle = st.number_input("Cycle number", min_value=1, value=max(1, int(medians.get("cycle", 100))))
            with c2:
                setting_1 = st.number_input("Setting 1", value=float(medians.get("setting_1", 0.0)), format="%.5f")
            with c3:
                setting_2 = st.number_input("Setting 2", value=float(medians.get("setting_2", 0.0)), format="%.5f")
            setting_3 = st.number_input("Setting 3", value=float(medians.get("setting_3", 0.0)), format="%.5f")

            st.markdown("#### Sensor measurements")
            st.caption("Defaults are medians from the training dataset when that file is available.")
            sensor_values = {}
            with st.expander("Adjust sensor readings", expanded=True):
                sensor_cols = st.columns(3)
                for i in range(1, 22):
                    key = f"sensor_{i}"
                    default_value = float(medians.get(key, 0.0))
                    with sensor_cols[(i - 1) % 3]:
                        sensor_values[key] = st.number_input(
                            f"Sensor {i}", value=default_value, format="%.5f", key=f"input_{key}"
                        )

            submitted = st.form_submit_button("Estimate RUL", use_container_width=True)

        if submitted:
            values = {
                "cycle": cycle, "setting_1": setting_1,
                "setting_2": setting_2, "setting_3": setting_3,
                **sensor_values
            }
            try:
                input_df = pd.DataFrame([[values[col] for col in feature_columns]], columns=feature_columns)
                prediction = max(0.0, float(model.predict(input_df)[0]))
                st.markdown("### Prediction")
                a, b = st.columns([1, 1])
                with a:
                    st.metric("Estimated RUL", f"{prediction:.1f} cycles")
                with b:
                    st.metric("Approximate cycles remaining", f"{prediction:.1f}")
                if prediction <= 20:
                    st.warning("The estimate is relatively low. Review maintenance requirements and inspect the engine.")
                else:
                    st.info("This is a model estimate, not a guarantee of safe operation.")
                with st.expander("Review model input"):
                    st.dataframe(input_df, use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# -------------------- PROJECT NOTES --------------------
elif page == "Project Notes":
    st.subheader("Project notes")
    st.markdown("""
    <div class="panel">
    <h3>Objective</h3>
    Estimate the Remaining Useful Life (RUL) of a turbofan engine from
    operating cycles, settings, and sensor measurements.
    <hr>
    <h3>Dataset</h3>
    NASA C-MAPSS FD001. Training RUL is constructed as each engine's
    final observed cycle minus its current cycle.
    <hr>
    <h3>Algorithms compared</h3>
    Linear Regression, Polynomial Regression, Decision Tree,
    Random Forest, and Gradient Boosting.
    <hr>
    <h3>Final model</h3>
    PolynomialFeatures (degree 2) + Ridge Regression, packaged in a
    scikit-learn pipeline.
    <hr>
    <h3>Metrics</h3>
    MAE, MSE, RMSE and R². Precision and recall are classification metrics
    and are not directly appropriate for this regression task.
    </div>
    """, unsafe_allow_html=True)
    st.warning(
        "Academic demonstration only. Do not use this prediction alone for real-world "
        "aviation maintenance or safety decisions."
    )

st.divider()
st.markdown(
    '<div class="small-note">AeroGuard · Predictive Maintenance with Machine Learning · NASA C-MAPSS FD001</div>',
    unsafe_allow_html=True
)
