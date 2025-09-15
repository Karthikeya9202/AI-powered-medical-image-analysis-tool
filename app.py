import streamlit as st
import cv2
from analysis.patient_analysis import analyze_medical_image
from analysis.doctor_analysis import analyze_doctor_images
from analysis.gradcam import generate_gradcam
from utils.image_utils import save_temp_file, cleanup_file

# -----------------------------
# 🔧 Page Config
# -----------------------------
st.set_page_config(page_title="Medical Image Analysis", layout="centered")

# -----------------------------
# 🏷️ Header & Tool Info
# -----------------------------
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown("""
This AI-powered tool helps in analyzing medical images.  
It can:
- Analyze patient scans
- Provide diagnostic insights
- Explain findings in simple terms
- Add research references
- Highlight regions of interest with heatmaps (Grad-CAM)
""")

# -----------------------------
# 🧑‍⚕️ Select Role
# -----------------------------
role = st.radio(
    "Select your role:",
    ("Patient", "Doctor"),
    index=None
)

# -----------------------------
# 🧍 Patient Workflow
# -----------------------------
if role == "Patient":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Medical Image", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.sidebar.button("🔍 Analyze Image"):
            with st.spinner("Analyzing the image..."):
                path = save_temp_file(uploaded_file)
                rgb_img, report = analyze_medical_image(path)

                if rgb_img is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(rgb_img, caption="Processed Image", use_container_width=True)
                    with col2:
                        cam_img = generate_gradcam(cv2.imread(path))
                        st.image(cam_img, caption="AI Attention Map (Grad-CAM)", use_container_width=True)

                st.subheader("📋 Analysis Report")
                st.markdown(report)
                cleanup_file(path)

# -----------------------------
# 👨‍⚕️ Doctor Workflow
# -----------------------------
elif role == "Doctor":
    uploaded_files = st.sidebar.file_uploader(
        "Upload patient scans", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True
    )

    if uploaded_files and st.sidebar.button("🔍 Analyze All Scans"):
        paths = [save_temp_file(f) for f in uploaded_files]

        with st.spinner("Analyzing scans..."):
            reports = analyze_doctor_images(paths)

            for idx, (img, report) in enumerate(reports, 1):
                st.image(img, caption=f"Processed Image {idx}", use_container_width=True)
                cam_img = generate_gradcam(cv2.imread(paths[idx - 1]))
                st.image(cam_img, caption=f"Grad-CAM Map {idx}", use_container_width=True)

                st.subheader(f"📋 Report {idx}")
                st.markdown(report)

        for p in paths:
            cleanup_file(p)
