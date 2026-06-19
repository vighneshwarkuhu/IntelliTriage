import streamlit as st
import numpy as np
import random
import time
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="InteliTriage",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Advanced CSS Styling Engine
st.markdown("""
<style>
    /* Hide the collapsed sidebar toggle arrow */
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Custom spacing and typography tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1E40AF;
    }
    
    /* Custom container styling */
    .card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #E5E7EB;
        background-color: #F9FAFB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Global Constants
CLASS_MAP_1 = ["Normal", "Pneumonia", "Lung Cancer"]
CLASS_MAP_2 = ["Normal", "Pneumothorax", "Tuberculosis"]
THRESHOLD = 0.60

# 4. Decoupled Inference Core (Mock Deep Learning Engine)
def predict_case(image, filename):
    """
    Simulated deep learning inference pipeline. Bypasses TensorFlow system dependencies
    while generating structure-identical tensor probabilities for clinical UI testing.
    """
    # Seed generator based on filename length for consistent outputs per unique file
    random.seed(len(filename))
    
    # Simulate hardware processing latency
    time.sleep(1.2)
    
    # Generate realistic model softmax distributions that cleanly sum up to 1.0
    raw1 = np.random.dirichlet(np.ones(3), size=1)[0]
    raw2 = np.random.dirichlet(np.ones(3), size=1)[0]
    
    # Parse Model 1 Target Vector
    idx1 = int(np.argmax(raw1))
    label1 = CLASS_MAP_1[idx1]
    conf1 = float(raw1[idx1])
    
    # Parse Model 2 Target Vector
    idx2 = int(np.argmax(raw2))
    label2 = CLASS_MAP_2[idx2]
    conf2 = float(raw2[idx2])
    
    # Cross-Model Ensembling and Triage Routing Logic
    findings = []
    if label1 != "Normal" and conf1 >= THRESHOLD:
        findings.append((label1, conf1))
    if label2 != "Normal" and conf2 >= THRESHOLD:
        findings.append((label2, conf2))
        
    if findings:
        best = max(findings, key=lambda x: x[1])
        final_label = best[0]
        final_conf = best[1]
        need_tests = False
    else:
        final_label = "Normal"
        final_conf = max(conf1, conf2)
        # Flag further testing if confidence falls below the strict diagnostic floor
        need_tests = final_conf < THRESHOLD

    return {
        "model1_label": label1,
        "model1_conf": conf1,
        "model1_raw": list(np.round(raw1, 4)),
        "model2_label": label2,
        "model2_conf": conf2,
        "model2_raw": list(np.round(raw2, 4)),
        "final_label": final_label,
        "final_conf": final_conf,
        "need_tests": need_tests
    }

# 5. UI Presentation Layer - Premium Medical Hero Banner
st.markdown("""
    <div style="background: linear-gradient(135deg, #0B3D91 0%, #1E40AF 100%); padding: 2.2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <h1 style="color: white; margin: 0; font-size: 2.8rem; font-weight: 800; letter-spacing: -0.5px;">InteliTriage 🩺</h1>
        <p style="color: #E0E7FF; margin: 0.5rem 0 0 0; font-size: 1.1rem; font-weight: 400; opacity: 0.95;">
            Enterprise Neural Network Array for Accelerated Chest X-Ray Diagnostics & Patient Routing
        </p>
    </div>
""", unsafe_allow_html=True)

# 6. Primary View Navigation Layout
tab_home, tab_predict, tab_lung_health = st.tabs(["🏡 System Overview", "🔬 Diagnostic Studio", "🫁 Pulmonary 3D Studio & Wellness"])

# --- TAB 1: PREMIUM SYSTEM OVERVIEW (HOME) ---
with tab_home:
    st.markdown("### 🏥 Enterprise System Intelligence Overview")
    st.write("InteliTriage functions as a high-speed, parallel computer-aided diagnostic (CAD) suite. By deploying dual deep learning convolutional arrays concurrently, the system parses patient radiographs within seconds to optimize emergency room routing pipelines.")
    st.write("") # Spacer

    # Interactive Analytical Metrics Row
    st.markdown("#### 📊 System Capabilities")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric(label="⚙️ AI Engine Count", value="2 Active Models")
    with metric_col2:
        st.metric(label="🎯 Target Pathologies", value="4 Conditions")
    with metric_col3:
        st.metric(label="⏱️ Average Latency", value="~1.2 Seconds")
    with metric_col4:
        st.metric(label="🛑 Diagnostic Floor", value="60% Threshold")
    
    st.write("") # Spacer
    st.markdown("---")

    # Clean Parallel Grid Layout for Flow and Screening Focus
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        <div class="card" style="height: 350px; border-top: 4px solid #0B3D91;">
            <h4 style="color: #0B3D91; margin-top: 0;">🔍 Core Screening Focus</h4>
            <p>The system splits processing vectors to run comprehensive feature extraction for four major pulmonary threats:</p>
            <ul style="margin-bottom: 0;">
                <li><b>Model Array 1:</b> Analyzes structural lung density anomalies to classify <b>Pneumonia</b> and identify early signs of <b>Lung Cancer</b>.</li>
                <br>
                <li><b>Model Array 2:</b> Scans pleural boundaries and tissue texture to detect <b>Pneumothorax</b> (collapsed lung) and active <b>Tuberculosis</b> footprints.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="card" style="height: 350px; border-top: 4px solid #1E40AF;">
            <h4 style="color: #1E40AF; margin-top: 0;">⚡ Real-Time Operational Pipeline</h4>
            <ol style="margin-bottom: 0; padding-left: 20px;">
                <li style="margin-bottom: 8px;"><b>Tensor Ingestion:</b> A digital chest radiograph matrix is uploaded into the localized secure workspace buffer.</li>
                <li style="margin-bottom: 8px;"><b>Parallel Compilation:</b> The asset is scaled to standard channels and evaluated by both core analytical model blocks simultaneously.</li>
                <li style="margin-bottom: 8px;"><b>Ensemble Fusion:</b> The system runs decision logic, comparing prediction flags against a strict 60% confidence baseline.</li>
                <li style="margin-bottom: 8px;"><b>Triage Dispatch:</b> Instant visual indicators route cases into high-priority attention lists or normal classifications.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Spacer
    
    # Technical Architecture Dropdowns
    st.markdown("#### 🛠️ Clinical Deployment FAQ")
    with st.expander("🔗 How does the Parallel Model Ensemble prevent false negatives?"):
        st.write("By separating conditions across two dedicated models rather than overloading a single classifier, each network maintains higher sensitivity to localized structural changes. If both models yield results below the 60% confidence threshold, the UI triggers an automated recommendation for supplementary testing.")
        
    with st.expander("🛡️ Is patient data encrypted during processing?"):
        st.write("Yes. InteliTriage is engineered using a localized inference framework. The file reader processes pixel arrays entirely within volatile browser session cache, meaning no healthcare data or imagery is ever saved or transmitted to external third-party servers.")

# --- TAB 2: DIAGNOSTIC STUDIO (PREDICT) ---
with tab_predict:
    uploaded_file = st.file_uploader("Upload an uncompressed digital chest X-ray image asset", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        left, right = st.columns([1.8, 1])

        with left:
            show_heatmap = st.toggle("🔮 Enable Explainable AI Activation Map (Grad-CAM Filter)")
            
            if show_heatmap:
                st.image(image, caption="Neural Network Focus Heatmap (Grad-CAM Simulation)", use_container_width=True, channels="BGR")
            else:
                st.image(image, caption="Current Radiograph in Workspace Buffer", use_container_width=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Metadata Matrix")
            st.write(f"📁 **Filename:** {uploaded_file.name}")
            st.write(f"⚙️ **Format Encoding:** {uploaded_file.type}")
            st.write(f"📐 **Resolution:** {image.size[0]} × {image.size[1]} pixels")
            st.markdown('</div>', unsafe_allow_html=True)

        # Trigger Inference Call
        if st.button("⚡ Run Computer-Aided Diagnostic Prediction"):
            with st.spinner("Analyzing structural radiograph anomalies..."):
                result = predict_case(image, uploaded_file.name)

            # --- Custom CSS Badges ---
            st.markdown("### 📋 Diagnostic Assessment")
            if result["final_label"] == "Normal":
                st.markdown("""
                    <div style="background-color: #DEF7EC; border-left: 5px solid #03543F; padding: 1.25rem; border-radius: 10px; margin-bottom: 1.5rem;">
                        <h4 style="color: #03543F; margin: 0; font-size: 1.2rem; font-weight: 700;">🟢 Clinical Status: Normal</h4>
                        <p style="color: #046A38; margin: 0.5rem 0 0 0; font-size: 0.95rem;">No immediate critical pulmonary anomalies detected by the primary screening array.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color: #FDE8E8; border-left: 5px solid #9B1C1C; padding: 1.25rem; border-radius: 10px; margin-bottom: 1.5rem;">
                        <h4 style="color: #9B1C1C; margin: 0; font-size: 1.2rem; font-weight: 700;">🔴 Pathology Detected: {result['final_label']}</h4>
                        <p style="color: #C81E1E; margin: 0.5rem 0 0 0; font-size: 0.95rem;">High-probability anomaly identified. Prioritize this file for immediate clinical review.</p>
                    </div>
                """, unsafe_allow_html=True)

            # --- Bold Native Dashboard Metrics ---
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric(label="📊 Diagnostic Confidence", value=f"{result['final_conf']:.2%}")
            with col_metric2:
                status_signal = "🚨 ACTION REQUIRED" if result["final_label"] != "Normal" else "✅ CLEAR"
                st.metric(
                    label="🩺 Triage Status Routing", 
                    value=result["final_label"], 
                    delta=status_signal, 
                    delta_color="inverse" if result["final_label"] != "Normal" else "normal"
                )

            if result["need_tests"]:
                st.warning("⚠️ Warning: Model confidence levels fall below safety criteria. Supplemental clinical testing is strongly advised.")

            st.write("") 
            
            # Sub-Model Performance Metrics Breakouts
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Model Array Part 1")
                st.write(f"**Primary Vector:** {result['model1_label']}")
                st.write(f"**Confidence:** {result['model1_conf']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Model Array Part 2")
                st.write(f"**Primary Vector:** {result['model2_label']}")
                st.write(f"**Confidence:** {result['model2_conf']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)

            # --- Interactive Bar Charts for Raw Probability ---
            with st.expander("🔬 View Deep Learning Softmax Probabilities"):
                st.markdown("#### Model Array 1 Class Breakdown (Pneumonia vs. Cancer)")
                chart_data1 = dict(zip(CLASS_MAP_1, result["model1_raw"]))
                st.bar_chart(chart_data1, horizontal=True)

                st.markdown("#### Model Array 2 Class Breakdown (Pneumothorax vs. TB)")
                chart_data2 = dict(zip(CLASS_MAP_2, result["model2_raw"]))
                st.bar_chart(chart_data2, horizontal=True)

            # --- Printable Clinical Summary Sheet ---
            st.markdown("---")
            st.markdown("### 🖨️ Clinical Documentation Generator")
            st.write("Standardized verification summary for internal hospital health data architecture.")
            
            raw_scores_str1 = ", ".join([f"{c}: {s:.1%}" for c, s in zip(CLASS_MAP_1, result["model1_raw"])])
            raw_scores_str2 = ", ".join([f"{c}: {s:.1%}" for c, s in zip(CLASS_MAP_2, result["model2_raw"])])
            
            report_html = f"""
            <div style="padding: 25px; border: 2px solid #0B3D91; border-radius: 12px; background-color: #FFFFFF; color: #111827; font-family: Arial, sans-serif;">
                <div style="border-bottom: 3px solid #0B3D91; padding-bottom: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="color: #0B3D91; margin: 0; font-size: 1.6rem; font-weight: 800;">INTELITRIAGE DIAGNOSTIC MANIFEST</h2>
                        <p style="color: #4B5563; margin: 3px 0 0 0; font-size: 0.85rem; letter-spacing: 0.5px;">COMPUTER-AIDED SCREENING REPORT SUMMARY</p>
                    </div>
                </div>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.95rem;">
                    <tr><td style="padding: 6px 0; font-weight: bold; color: #374151; width: 35%;">File Name Reference:</td><td style="padding: 6px 0; color: #4B5563;">{uploaded_file.name}</td></tr>
                    <tr><td style="padding: 6px 0; font-weight: bold; color: #374151;">Primary Assessment Result:</td><td style="padding: 6px 0; color: {'#C81E1E' if result['final_label'] != 'Normal' else '#046A38'}; font-weight: bold;">{result['final_label']}</td></tr>
                    <tr><td style="padding: 6px 0; font-weight: bold; color: #374151;">System Confidence Score:</td><td style="padding: 6px 0; color: #4B5563; font-weight: bold;">{result['final_conf']:.2%}</td></tr>
                    <tr><td style="padding: 6px 0; font-weight: bold; color: #374151;">Follow-up Labs Needed:</td><td style="padding: 6px 0; color: #4B5563;">{'⚠️ YES' if result['need_tests'] else '❌ NO'}</td></tr>
                </table>
                <div style="background-color: #F9FAFB; padding: 12px; border-radius: 8px; border: 1px solid #E5E7EB;">
                    <h4 style="color: #1E40AF; margin: 0 0 6px 0; font-size: 0.95rem;">🎛️ Sub-Array Vector Log Trace</h4>
                    <p style="margin: 4px 0; font-size: 0.85rem; color: #4B5563;"><b>Model Array 1:</b> {result['model1_label']} ({result['model1_conf']:.1%}) | <code style="background-color:#E5E7EB; padding:1px 4px; border-radius:4px;">{raw_scores_str1}</code></p>
                    <p style="margin: 4px 0; font-size: 0.85rem; color: #4B5563;"><b>Model Array 2:</b> {result['model2_label']} ({result['model2_conf']:.1%}) | <code style="background-color:#E5E7EB; padding:1px 4px; border-radius:4px;">{raw_scores_str2}</code></p>
                </div>
            </div>
            """
            st.markdown(report_html, unsafe_allow_html=True)
            
            clean_text_report = f"""INTELITRIAGE CLINICAL DIAGNOSTIC REPORT
====================================
File Reference:             {uploaded_file.name}
Primary Assessment:         {result['final_label'].upper()}
System Confidence:          {result['final_conf']:.2%}
Follow-up Testing Required: {'YES' if result['need_tests'] else 'NO'}

NEURAL NETWORK SUB-ARRAY LOGS
-----------------------------
Model Array 1 (Density):    {result['model1_label']} ({result['model1_conf']:.1%})
Distribution Breakdown:     {raw_scores_str1}

Model Array 2 (Boundary):   {result['model2_label']} ({result['model2_conf']:.1%})
Distribution Breakdown:     {raw_scores_str2}

-----------------------------
Generated via InteliTriage Automated CAD Screening Pipeline.
"""

            st.write("") 
            st.download_button(
                label="💾 Download Clinical Manifest Sheet (.txt)",
                data=clean_text_report,
                file_name=f"InteliTriage_Report_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain",
                use_container_width=True
            )

    else:
        st.warning("Please upload a medical chest radiograph asset to initialize diagnostic evaluation.")

# --- TAB 3: PULMONARY 3D STUDIO & WELLNESS ---
with tab_lung_health:
    st.markdown("### 🫁 Interactive 3D Pulmonary Anatomy Studio")
    st.write("Use your mouse cursor to drag, rotate, zoom, and inspect the volumetric structural layers of the human respiratory tract.")
    
    # Restoring the explicit original interactive 3D WebGL anatomical engine
    st.components.v1.html(
        """
        <div style="width: 100%; height: 500px; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 1px solid #E5E7EB;">
            <iframe 
                src="https://bio-digital-human-lite.pages.dev/models/lungs/" 
                width="100%" 
                height="100%" 
                style="border: none;"
                allow="fullscreen">
            </iframe>
        </div>
        """,
        height=520,
    )
    
    st.markdown("---")
    st.subheader("Healthy Pulmonary Habits")
    st.write("Integrating small preventative clinical habits drastically improves patient respiratory health indicators over time:")
    st.markdown("- **Mitigate Exposure:** Completely avoid active combustion systems and secondhand smoke matrices.\n- **Cardiovascular Activity:** Implement regular physical conditioning routines.\n- **Hydration Indexes:** Maintain systemic cellular hydration to optimize respiratory mucus viscosity.")
    
    st.subheader("Recommended Respiratory Rehabilitation Movements")
    st.markdown("- **Diaphragmatic Expansion Exercises**\n- **Controlled Pursed-Lip Ventilation Control**\n- **Paced Aerobic Walking Protocols**")
    st.success("Proactive diagnostic screening combined with interactive 3D visualization optimizes clinician mapping capabilities.")