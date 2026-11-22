import json
import re
import time
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ThermoHand AI | Joint Thermal Analyzer",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Light Clinical Teal & Crisp Slate Theme (High-Contrast Text)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Soft light canvas + high-contrast dark text everywhere */
        .stApp {
            background-color: #F4F7FA;
            color: #0F172A;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }

        /* Hide default Streamlit menu/footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background-color: transparent !important;}

        /* Force crisp dark text on all Streamlit labels, radios, and markdown */
        .stMarkdown, .stMarkdown p, label, .stRadio label, .stSelectbox label,
        .stTextInput label, .stNumberInput label, .stSlider label,
        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stCaptionContainer"] p {
            color: #1E293B !important;
        }
        div[data-testid="stCaptionContainer"] p {
            color: #475569 !important;
        }

        /* Light Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 60%, #E0F2FE 100%);
            border: 1px solid #CBD5E1;
            border-left: 6px solid #0F766E;
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 1.4rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }
        .hero-badge {
            display: inline-block;
            background: #CCFBF1;
            color: #0F766E;
            border: 1px solid #99F6E4;
            border-radius: 999px;
            padding: 3px 12px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            color: #0F172A !important;
            margin: 0;
        }
        .hero-sub {
            color: #475569 !important;
            font-size: 0.96rem;
            margin-top: 6px;
            margin-bottom: 0;
        }

        /* Pure White Cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF;
            border: 1px solid #DCE4EE !important;
            border-radius: 14px !important;
            padding: 8px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        /* Section Tag */
        .section-tag {
            color: #0F766E;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        /* Inputs & Selectboxes */
        .stTextInput input, .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }

        /* File Uploader Dropzones */
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #F8FAFC !important;
            border: 1.5px dashed #94A3B8 !important;
            border-radius: 10px !important;
            color: #1E293B !important;
        }
        section[data-testid="stFileUploaderDropzone"] * {
            color: #1E293B !important;
        }
        section[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #0F766E !important;
            background-color: #F0FDFA !important;
        }

        /* Primary Teal Button */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0F766E 0%, #0284C7 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            padding: 0.65rem 1.2rem !important;
            box-shadow: 0 4px 12px rgba(15, 118, 110, 0.25);
        }
        div.stButton > button[kind="primary"]:hover {
            opacity: 0.94;
        }
        div.stButton > button[kind="primary"]:disabled {
            background: #E2E8F0 !important;
            color: #94A3B8 !important;
            box-shadow: none !important;
        }

        /* KPI Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #DCE4EE;
            border-top: 4px solid #0F766E;
            padding: 14px 18px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] p {
            color: #475569 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #0F172A !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: #475569 !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #0F766E !important;
        }

        /* Privacy Notice */
        .privacy-strip {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 10px;
            padding: 10px 14px;
            color: #1E3A8A;
            font-size: 0.85rem;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "wmv"]


# -----------------------------------------------------------------------------
# Pipeline Hook (Wire in your MediaPipe + Radiometric Thermal model here)
# -----------------------------------------------------------------------------
def run_hand_temp_pipeline(
    participant_id: str,
    protocol: str,
    rgb_file,
    thermal_file,
    radiometric_files,
    sample_every_sec: float,
    calibration_file,
    adv_config: dict,
    status_box,
):
    status_box.write("1/4 Loading RGB, thermal, and radiometric arrays...")
    time.sleep(0.5)

    if calibration_file is None:
        status_box.write("2/4 Estimating RGB–Thermal rig calibration from recording...")
        time.sleep(0.7)
        rig_calibration = {
            "participant_id": participant_id,
            "homography_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "note": "Generated from recording",
        }
    else:
        status_box.write("2/4 Applying uploaded rig_calibration.json...")
        rig_calibration = json.load(calibration_file)
        time.sleep(0.4)

    status_box.write(
        f"3/4 Running MediaPipe hand landmarks (conf ≥ {adv_config['mp_confidence']:.2f}, "
        f"interval {sample_every_sec:.2f}s)..."
    )
    time.sleep(0.7)

    status_box.write("4/4 Computing radiometric joint temperatures & ΔT fluctuations...")
    time.sleep(0.5)

    timestamps = np.arange(0, 60, max(sample_every_sec, 1.0))
    rng = np.random.default_rng(42)
    joints = [
        "Wrist",
        "Thumb_CMC",
        "Index_MCP",
        "Index_PIP",
        "Middle_MCP",
        "Middle_PIP",
        "Ring_PIP",
        "Pinky_TIP",
    ]

    time_series = pd.DataFrame({"Time (s)": timestamps})
    summary_rows = []
    offset = 273.15 if adv_config["temp_unit"] == "K" else 0.0
    unit = adv_config["temp_unit"]

    for j in joints:
        base = rng.uniform(31.2, 34.4)
        curve = base + np.cumsum(rng.normal(0.02, 0.12, size=len(timestamps))) + offset
        time_series[j] = np.round(curve, 2)
        summary_rows.append(
            {
                "Joint": j,
                f"Mean ({unit})": round(float(np.mean(curve)), 2),
                f"Min ({unit})": round(float(np.min(curve)), 2),
                f"Max ({unit})": round(float(np.max(curve)), 2),
                f"Δ Temp ({unit})": round(float(np.max(curve) - np.min(curve)), 2),
                f"Std Dev ({unit})": round(float(np.std(curve)), 2),
            }
        )

    return {
        "participant_id": participant_id,
        "protocol": protocol,
        "unit": unit,
        "frames_processed": len(timestamps),
        "summary_df": pd.DataFrame(summary_rows),
        "time_series": time_series,
        "rig_calibration": rig_calibration,
    }


# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">Multimodal RGB + FLIR Radiometric Pipeline</div>
        <h1 class="hero-title">ThermoHand AI — Hand Joint Thermal Analyzer</h1>
        <p class="hero-sub">
            MediaPipe 21-point landmark tracking fused with radiometric thermal arrays for joint temperature and fluctuation analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Split-Screen Workspace (Left: Session & Config | Right: Media Uploads)
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.05, 1.55], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="section-tag">Step 01 • Session & Calibration</div>', unsafe_allow_html=True)

        p_col, proto_col = st.columns([1.4, 1])
        with p_col:
            participant_id = st.text_input(
                "Participant ID",
                placeholder="e.g., S1_open_close",
            ).strip()
        with proto_col:
            protocol = st.selectbox("Protocol", ["open_close", "static", "custom"])

        id_valid = bool(participant_id) and bool(ID_PATTERN.match(participant_id))
        if participant_id and not id_valid:
            st.error("Use only letters, digits, '-' and '_'.")

        sample_every_sec = st.number_input(
            "Sampling interval (seconds / frame)",
            min_value=0.1,
            max_value=60.0,
            value=5.00,
            step=0.5,
            format="%.2f",
        )

        cal_mode = st.radio(
            "Camera rig calibration",
            options=["Calibrate from this recording", "Upload rig_calibration.json"],
            horizontal=True,
        )

        calibration_file = None
        if cal_mode == "Upload rig_calibration.json":
            calibration_file = st.file_uploader("Upload rig_calibration.json", type=["json"])

        with st.expander("Model & ROI parameters"):
            mp_confidence = st.slider("MediaPipe confidence", 0.1, 0.95, 0.50, 0.05)
            roi_radius = st.number_input("Thermal ROI radius (px)", min_value=1, max_value=25, value=5)
            temp_unit = st.selectbox("Temperature unit", ["°C", "K"])
            adv_config = {
                "mp_confidence": mp_confidence,
                "roi_radius": roi_radius,
                "temp_unit": temp_unit,
            }

        st.markdown(
            """
            <div class="privacy-strip">
                🔒 <strong>Ephemeral session:</strong> Recordings and outputs exist only while this tab is open
                and are cleared on refresh. Download your CSVs and calibration file before closing.
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    with st.container(border=True):
        st.markdown('<div class="section-tag">Step 02 • Multimodal Data Ingestion</div>', unsafe_allow_html=True)

        u1, u2 = st.columns(2, gap="medium")
        with u1:
            st.markdown("**RGB Video**")
            rgb_file = st.file_uploader(
                "RGB Video",
                type=VIDEO_TYPES,
                label_visibility="collapsed",
                key="rgb_u",
            )
        with u2:
            st.markdown("**FLIR Thermal Video**")
            thermal_file = st.file_uploader(
                "Thermal Video",
                type=VIDEO_TYPES,
                label_visibility="collapsed",
                key="th_u",
            )

        st.markdown("**Radiometric Array (.npy)**")
        rad_mode = st.radio(
            "Radiometric layout",
            options=["Single stacked .npy", "Multiple .npy (one per frame)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        allow_multi = rad_mode == "Multiple .npy (one per frame)"
        radiometric_files = st.file_uploader(
            "Radiometric .npy",
            type=["npy"],
            accept_multiple_files=allow_multi,
            label_visibility="collapsed",
            key="rad_u",
        )

        # Validation & Run Trigger
        has_rad = bool(radiometric_files) if allow_multi else (radiometric_files is not None)
        cal_ready = (cal_mode == "Calibrate from this recording") or (calibration_file is not None)
        ready_to_run = all([id_valid, rgb_file is not None, thermal_file is not None, has_rad, cal_ready])

        missing = []
        if not id_valid:
            missing.append("Participant ID")
        if rgb_file is None:
            missing.append("RGB video")
        if thermal_file is None:
            missing.append("Thermal video")
        if not has_rad:
            missing.append("Radiometric .npy")
        if not cal_ready:
            missing.append("Calibration JSON")

        st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)
        btn_col, msg_col = st.columns([1, 1.8])
        with btn_col:
            run_clicked = st.button(
                "Run Thermal Analysis",
                type="primary",
                disabled=not ready_to_run,
                use_container_width=True,
            )
        with msg_col:
            if missing:
                st.caption(f"Pending inputs: {', '.join(missing)}")
            else:
                st.caption("All inputs verified. Ready to run.")


# -----------------------------------------------------------------------------
# Execution & Results Dashboard
# -----------------------------------------------------------------------------
if run_clicked:
    with st.status("Processing multimodal hand recordings...", expanded=True) as status_box:
        results = run_hand_temp_pipeline(
            participant_id=participant_id,
            protocol=protocol,
            rgb_file=rgb_file,
            thermal_file=thermal_file,
            radiometric_files=radiometric_files,
            sample_every_sec=sample_every_sec,
            calibration_file=calibration_file,
            adv_config=adv_config,
            status_box=status_box,
        )
        status_box.update(label="Thermal analysis complete", state="complete", expanded=False)
    st.session_state["analysis_results"] = results

if "analysis_results" in st.session_state:
    res = st.session_state["analysis_results"]
    summary_df = res["summary_df"]
    ts_df = res["time_series"]
    unit = res["unit"]
    mean_col = f"Mean ({unit})"
    delta_col = f"Δ Temp ({unit})"

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    st.markdown(f"### Analysis Dashboard — `{res['participant_id']}` ({res['protocol']})")

    k1, k2, k3, k4 = st.columns(4)
    hottest = summary_df.loc[summary_df[mean_col].idxmax()]
    max_fluct = summary_df.loc[summary_df[delta_col].idxmax()]

    k1.metric("Frames Sampled", res["frames_processed"])
    k2.metric("Overall Mean Temp", f"{summary_df[mean_col].mean():.2f} {unit}")
    k3.metric("Warmest Joint", hottest["Joint"], f"{hottest[mean_col]:.2f} {unit}")
    k4.metric("Peak Fluctuation", max_fluct["Joint"], f"Δ {max_fluct[delta_col]:.2f} {unit}")

    tab_summary, tab_chart, tab_visuals, tab_export = st.tabs(
        ["Joint Metrics", "Fluctuation Curves", "RGB / Thermal Overlay", "Export Files"]
    )

    with tab_summary:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tab_chart:
        picked = st.multiselect(
            "Compare hand joints over time",
            options=list(summary_df["Joint"]),
            default=list(summary_df["Joint"])[:4],
        )
        if picked:
            st.line_chart(ts_df.set_index("Time (s)")[picked], height=360)

    with tab_visuals:
        v1, v2 = st.columns(2)
        with v1:
            with st.container(border=True):
                st.markdown("**RGB + MediaPipe 21-Point Landmarks**")
                st.caption("Connect your annotated RGB frame here via `st.image(...)`.")
        with v2:
            with st.container(border=True):
                st.markdown("**Aligned Radiometric Thermal Map**")
                st.caption("Connect your aligned FLIR thermal ROI frame here via `st.image(...)`.")

    with tab_export:
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "Joint Summary (.csv)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{res['participant_id']}_joint_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with e2:
            st.download_button(
                "Time Series (.csv)",
                data=ts_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{res['participant_id']}_timeseries.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with e3:
            st.download_button(
                "rig_calibration.json",
                data=json.dumps(res["rig_calibration"], indent=2).encode("utf-8"),
                file_name="rig_calibration.json",
                mime="application/json",
                use_container_width=True,
            )

    