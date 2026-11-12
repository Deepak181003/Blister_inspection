"""
app.py
======
Interactive Web Browser Interface for Automated Pharmaceutical Blister Pack Defect Detection.
Built with Streamlit and OpenCV.

Run command:
  streamlit run app.py
"""

import os
import io
import time
import glob
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import (
    to_grayscale,
    apply_gaussian_blur,
    apply_adaptive_threshold,
    apply_canny_edge,
    apply_morphology,
    preprocess_pipeline
)
from detection import (
    inspect_blister_pack,
    BlisterPackReference,
    extract_pack_boundary,
    locate_cavity_rois,
    inspect_single_cavity,
    inspect_blister_seals,
    annotate_defects,
    InspectionResult
)
from report import log_inspection_result, DEFAULT_REPORT_PATH
from generate_dataset import build_full_dataset

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaVision - Blister Pack Inspection",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Pharma UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #E5E7EB;
    }
    .status-pass {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-reject {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-review {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def load_image_from_bytes(uploaded_file) -> np.ndarray:
    """Decodes uploaded file bytes into an OpenCV BGR image."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def run_cv_inspection(image: np.ndarray,
                       filename: str,
                       ref: BlisterPackReference) -> InspectionResult:
    """Executes the CV inspection engine on a NumPy image array."""
    start_time = time.perf_counter()

    # Step 1: Pack boundary and seal check
    pack_bbox, pack_contour, card_thresh = extract_pack_boundary(image)
    seal_defects = inspect_blister_seals(image, pack_bbox, pack_contour, card_thresh)

    # Step 2: Cavity grid ROIs
    cavity_rois = locate_cavity_rois(image, pack_bbox, ref)

    # Step 3: Inspect each cavity
    all_defects = list(seal_defects)
    cavity_metrics = []

    for idx, roi in enumerate(cavity_rois, start=1):
        defect, metrics = inspect_single_cavity(idx, image, roi, ref)
        cavity_metrics.append(metrics)
        if defect is not None:
            all_defects.append(defect)

    # Overall decision logic
    has_reject = any(d.decision == 'REJECT' for d in all_defects)
    has_review = any(d.decision == 'REVIEW' for d in all_defects)

    if has_reject:
        overall_decision = 'REJECT'
    elif has_review:
        overall_decision = 'REVIEW'
    else:
        overall_decision = 'PASS'

    # Annotation
    annotated = annotate_defects(image, cavity_rois, all_defects, overall_decision)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    summary = f"Inspected {len(cavity_rois)} cavities. Detected {len(all_defects)} defect(s)."

    return InspectionResult(
        image_filename=filename,
        overall_decision=overall_decision,
        total_defects=len(all_defects),
        defects=all_defects,
        cavity_rois=cavity_rois,
        processing_time_ms=round(elapsed_ms, 2),
        annotated_image=annotated,
        summary_notes=summary
    )


# ---------------------------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=64)
    st.title("PharmaVision AI")
    st.markdown("**Automated Blister Inspection**")
    st.markdown("---")

    app_mode = st.radio(
        "Navigation",
        ["🔍 Single Pack Inspection", "📊 Batch Quality Analytics", "⚙️ Dataset Generator"]
    )

    st.markdown("---")
    st.subheader("Inspection Parameters")
    with st.expander("Tuning Tolerances", expanded=False):
        min_fill = st.slider("Min Fill Ratio (Empty Cutoff)", 0.05, 0.40, 0.15, 0.01)
        max_edge = st.slider("Max Internal Crack Edges", 0.01, 0.08, 0.025, 0.005)
        ar_tol = st.slider("Aspect Ratio Tolerance", 0.10, 0.60, 0.35, 0.05)
        min_solidity = st.slider("Min Pill Solidity", 0.70, 0.99, 0.90, 0.01)

    ref_profile = BlisterPackReference(
        min_fill_ratio=min_fill,
        max_internal_edge_ratio=max_edge,
        max_aspect_ratio_deviation=ar_tol,
        min_solidity=min_solidity
    )


# ---------------------------------------------------------------------------
# MODE 1: Single Pack Inspection
# ---------------------------------------------------------------------------
if app_mode == "🔍 Single Pack Inspection":
    st.markdown('<div class="main-title">💊 Pharmaceutical Blister Pack Defect Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time computer vision quality control powered by OpenCV</div>', unsafe_allow_html=True)

    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        st.subheader("1. Select or Upload Image")
        input_source = st.radio("Choose Input Method:", ["Select Pre-loaded Sample", "Upload From Computer"], horizontal=True)

        selected_image = None
        source_name = ""

        if input_source == "Select Pre-loaded Sample":
            samples = sorted(glob.glob("dataset/*/*.png"))
            if not samples:
                st.warning("No sample images found. Building sample dataset now...")
                build_full_dataset()
                samples = sorted(glob.glob("dataset/*/*.png"))

            # Group samples nicely
            sample_labels = [f"{os.path.basename(os.path.dirname(s))} / {os.path.basename(s)}" for s in samples]
            selected_idx = st.selectbox("Choose a sample test case:", range(len(samples)), format_func=lambda i: sample_labels[i])
            chosen_path = samples[selected_idx]
            selected_image = cv2.imread(chosen_path)
            source_name = os.path.basename(chosen_path)

        else:
            uploaded_file = st.file_uploader("Upload blister pack image", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded_file is not None:
                selected_image = load_image_from_bytes(uploaded_file)
                source_name = uploaded_file.name

    with col_input2:
        if selected_image is not None:
            st.subheader("Preview Input Image")
            st.image(cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Selected: {source_name}")

    if selected_image is not None:
        st.markdown("---")
        # Run Inspection
        result = run_cv_inspection(selected_image, source_name, ref_profile)
        log_inspection_result(result)

        # -------------------------------------------------------------
        # Metrics Display
        # -------------------------------------------------------------
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        with mcol1:
            if result.overall_decision == "PASS":
                st.markdown('<div class="metric-card"><h4>DECISION</h4><span class="status-pass">PASS</span></div>', unsafe_allow_html=True)
            elif result.overall_decision == "REJECT":
                st.markdown('<div class="metric-card"><h4>DECISION</h4><span class="status-reject">REJECT</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h4>DECISION</h4><span class="status-review">REVIEW</span></div>', unsafe_allow_html=True)

        with mcol2:
            st.markdown(f'<div class="metric-card"><h4>DEFECT COUNT</h4><h2 style="margin:0; color:#1F2937;">{result.total_defects}</h2></div>', unsafe_allow_html=True)

        with mcol3:
            st.markdown(f'<div class="metric-card"><h4>LATENCY</h4><h2 style="margin:0; color:#2563EB;">{result.processing_time_ms} ms</h2></div>', unsafe_allow_html=True)

        with mcol4:
            st.markdown(f'<div class="metric-card"><h4>CAVITIES</h4><h2 style="margin:0; color:#4B5563;">{len(result.cavity_rois)} Pockets</h2></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # Defect Details Table
        # -------------------------------------------------------------
        if result.defects:
            st.error(f"⚠️ **{result.total_defects} Defect(s) Detected! Action Required: REJECT Pack.**")
            defect_records = []
            for d in result.defects:
                defect_records.append({
                    "Target": f"Cavity #{d.cavity_id}" if d.cavity_id < 900 else "Foil Seal Margin",
                    "Defect Category": d.defect_type,
                    "Decision": d.decision,
                    "Confidence": f"{d.confidence * 100:.1f}%",
                    "Diagnostic Explanation": d.description,
                    "Bounding Box (x,y,w,h)": str(d.bbox)
                })
            st.dataframe(pd.DataFrame(defect_records), use_container_width=True)
        else:
            st.success("✅ **Quality Control Passed:** All 10 cavities and outer blister seals conform to pharmaceutical specifications.")

        st.markdown("---")

        # -------------------------------------------------------------
        # Computer Vision Stages & Visualization
        # -------------------------------------------------------------
        st.subheader("Visual Inspection & OpenCV Pipeline Breakdown")

        tab_annotated, tab_stages, tab_cavities = st.tabs(["🎯 Annotated Result", "🔬 5-Panel CV Pipeline", "🔍 Cavity Pocket Grid"])

        with tab_annotated:
            st.image(
                cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Final Annotated Inspection: {result.overall_decision} ({result.total_defects} defect(s))"
            )

            # Download Annotated Image
            _, buf = cv2.imencode(".png", result.annotated_image)
            st.download_button(
                label="📥 Download Annotated Inspection Image",
                data=buf.tobytes(),
                file_name=f"annotated_{source_name}",
                mime="image/png"
            )

        with tab_stages:
            stages = preprocess_pipeline(selected_image)

            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.markdown("**1. Original BGR Image**")
                st.image(cv2.cvtColor(stages["original"], cv2.COLOR_BGR2RGB), use_container_width=True)
            with pcol2:
                st.markdown("**2. Grayscale Luminance**")
                st.image(stages["gray"], use_container_width=True)
            with pcol3:
                st.markdown("**3. Noise-Filtered Gaussian Blur**")
                st.image(stages["blurred"], use_container_width=True)

            pcol4, pcol5 = st.columns(2)
            with pcol4:
                st.markdown("**4. Adaptive / Morphological Threshold**")
                st.image(stages["thresh"], use_container_width=True)
            with pcol5:
                st.markdown("**5. Canny Edge Gradient Map**")
                st.image(stages["edges"], use_container_width=True)

        with tab_cavities:
            st.markdown("**Individual Pocket Regions of Interest (ROIs):**")
            cavity_cols = st.columns(5)
            defect_dict = {d.cavity_id: d for d in result.defects}

            for idx, (x, y, w, h) in enumerate(result.cavity_rois, start=1):
                col_idx = (idx - 1) % 5
                with cavity_cols[col_idx]:
                    crop = selected_image[y:y+h, x:x+w]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    if idx in defect_dict:
                        st.image(crop_rgb, caption=f"#{idx} ❌ {defect_dict[idx].defect_type}", use_container_width=True)
                    else:
                        st.image(crop_rgb, caption=f"#{idx} ✅ PASS", use_container_width=True)


# ---------------------------------------------------------------------------
# MODE 2: Batch Quality Analytics
# ---------------------------------------------------------------------------
elif app_mode == "📊 Batch Quality Analytics":
    st.markdown('<div class="main-title">📊 Batch Inspection & Quality Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Continuous automated batch inspection metrics & audit history</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Complete Batch Inspection on 'dataset/'", type="primary"):
        with st.spinner("Processing all blister pack test sets..."):
            all_images = sorted(glob.glob("dataset/*/*.png"))
            progress_bar = st.progress(0)

            for i, img_path in enumerate(all_images):
                res = inspect_blister_pack(img_path, reference=ref_profile)
                log_inspection_result(res)
                progress_bar.progress((i + 1) / len(all_images))

        st.success(f"Batch completed! Processed {len(all_images)} blister packs.")

    if os.path.exists(DEFAULT_REPORT_PATH):
        df = pd.read_csv(DEFAULT_REPORT_PATH)

        if not df.empty:
            st.markdown("### Executive Summary")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            total_inspected = len(df)
            decision_counts = df["Inspection_Decision"].value_counts().to_dict()
            passed = decision_counts.get("PASS", 0)
            rejected = decision_counts.get("REJECT", 0)
            pass_rate = (passed / total_inspected) * 100.0 if total_inspected > 0 else 0.0
            avg_time = df["Processing_Time_ms"].mean()

            kpi1.metric("Total Packs Inspected", total_inspected)
            kpi2.metric("Pass Rate", f"{pass_rate:.1f}%")
            kpi3.metric("Defects Discovered", rejected)
            kpi4.metric("Avg Latency", f"{avg_time:.2f} ms")

            st.markdown("---")
            st.markdown("### Defect Category Distribution")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                dec_df = df["Inspection_Decision"].value_counts().reset_index()
                dec_df.columns = ["Decision", "Count"]
                st.bar_chart(dec_df.set_index("Decision"))

            with chart_col2:
                cat_series = df[df["Defect_Category"] != "NONE"]["Defect_Category"]
                if not cat_series.empty:
                    split_cats = []
                    for item in cat_series:
                        split_cats.extend([c.strip() for c in item.split(";")])
                    cat_df = pd.Series(split_cats).value_counts().reset_index()
                    cat_df.columns = ["Defect Category", "Count"]
                    st.dataframe(cat_df, use_container_width=True)
                else:
                    st.info("No defects logged yet.")

            st.markdown("---")
            st.markdown("### Complete Inspection Audit Log")
            st.dataframe(df.sort_index(ascending=False), use_container_width=True)

            # CSV Download
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full CSV Inspection Audit Report",
                data=csv_data,
                file_name="blister_inspection_audit_report.csv",
                mime="text/csv"
            )
    else:
        st.info("No inspection reports available yet. Run a single or batch inspection to generate records.")


# ---------------------------------------------------------------------------
# MODE 3: Dataset Generator
# ---------------------------------------------------------------------------
elif app_mode == "⚙️ Dataset Generator":
    st.markdown('<div class="main-title">⚙️ Synthetic Blister Pack Dataset Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Synthesize realistic pharmaceutical blister packs with known ground-truth defects</div>', unsafe_allow_html=True)

    st.info("This utility creates realistic, high-fidelity synthetic pharmaceutical blister packs across all 5 standard categories: normal, missing capsules, cracked tablets, broken foil seals, and deformed cavities.")

    if st.button("Generate / Refresh All 10 Test Images", type="primary"):
        with st.spinner("Generating synthetic blister pack images..."):
            build_full_dataset("dataset")
        st.success("Successfully generated 10 test images in 'dataset/' subfolders!")

    st.markdown("### Sample Gallery")
    sample_categories = ["normal", "missing_capsule", "cracked_pill", "broken_seal", "deformed_cavity"]
    gallery_cols = st.columns(5)

    for i, cat in enumerate(sample_categories):
        img_path = os.path.join("dataset", cat, f"blister_{cat}_01.png")
        if os.path.exists(img_path):
            with gallery_cols[i]:
                st.image(img_path, caption=cat.replace("_", " ").title(), use_container_width=True)
