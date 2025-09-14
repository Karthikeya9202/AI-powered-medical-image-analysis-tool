import os
import cv2
import numpy as np
import streamlit as st
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# -----------------------------
# 🔑 API Key Setup
# -----------------------------
GOOGLE_API_KEY = "AIzaSyDNIpxHfk3bI95KbLzyDb7UWbM6cPHI"
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set GOOGLE_API_KEY as an environment variable or replace YOUR_API_KEY_HERE.")
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Vision Model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# DuckDuckGo Search
search_tool = DuckDuckGoSearchRun()

# -----------------------------
# 📝 Medical Analysis Prompt
# -----------------------------
query_template = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology. 
Analyze the uploaded medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.
- Evaluate image quality.

### 2. Key Findings
- Highlight observations systematically.
- Identify abnormalities, measurements, densities.

### 3. Diagnostic Assessment
- Primary diagnosis (with confidence level).
- Differential diagnoses.

### 4. Patient-Friendly Explanation
- Simplify findings in plain language.

### 5. Research Context
- Use DuckDuckGo search to provide 2–3 supporting references.
"""

# -----------------------------
# 🔬 Medical Image Analysis
# -----------------------------
def analyze_medical_image(image_path: str):
    try:
        # Read & resize image with OpenCV
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized = cv2.resize(image, (new_width, new_height))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert image to bytes for Gemini
        _, buffer = cv2.imencode(".png", resized)
        image_bytes = buffer.tobytes()

        # 🧠 Step 1: Get initial report (no research yet)
        draft_response = gemini_model.generate_content(
            [
                query_template,
                {"mime_type": "image/png", "data": image_bytes}
            ]
        )
        draft_text = draft_response.text

        # 🧠 Step 2: Ask Gemini for best search queries
        query_extraction_prompt = f"""
From this draft medical report:

{draft_text}

Generate 3 short search queries (max 10 words each) that would help
find supporting medical research. Only return queries, one per line.
"""
        queries_response = gemini_model.generate_content(query_extraction_prompt)
        queries = [q.strip() for q in queries_response.text.split("\n") if q.strip()]

        # 🔍 Step 3: Run DuckDuckGo for each query
        search_results = "\n".join(
            [f"**{q}:** {search_tool.run(q)}" for q in queries]
        )

        # 🧠 Step 4: Refine report with real references
        enriched_prompt = f"""
Here is the draft medical report:

{draft_text}

Now refine ONLY the "Research Context" section by integrating
the following DuckDuckGo search results:

{search_results}

Return the full updated report.
"""
        final_response = gemini_model.generate_content(enriched_prompt)

        return rgb_image, final_response.text

    except Exception as e:
        return None, f"⚠️ Analysis error: {e}"

cnn_model = ResNet50(weights="imagenet")

def generate_gradcam(image_bgr):
    """Generate a Grad-CAM heatmap for an input medical image (TensorFlow version)."""
    img_resized = cv2.resize(image_bgr, (224, 224))
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)

    grad_model = tf.keras.models.Model(
        [cnn_model.inputs], [cnn_model.get_layer("conv5_block3_out").output, cnn_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Resize heatmap to original image size
    heatmap = np.array(heatmap)  # ensures it's numpy
    heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Overlay heatmap on image
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_color, 0.4, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


doctor_query_template = """
You are a senior radiologist assisting another doctor.
Analyze the uploaded medical scan(s) and structure your response as follows:

### 1. Case Summary
- Imaging modality and anatomical region.
- Number of images provided.
- Overall image quality.

### 2. Findings per Image
- List abnormalities or notable features for each image.
- Give a severity score (Mild/Moderate/Severe).
- Highlight if the case looks emergency or routine.

### 3. Diagnostic Assessment
- Most likely diagnosis.
- Confidence level.
- Differential diagnoses (if applicable).

### 4. Clinical Recommendations
- Suggested next steps (lab tests, follow-up scans, treatment urgency).
- Mention if immediate intervention is needed.

### 5. Research References
- Summarize 2–3 relevant studies or guidelines (integrated via search).
"""

def analyze_doctor_images(image_paths: list):
    results = []

    for path in image_paths:
        # Load + preprocess
        image = cv2.imread(path)
        resized = cv2.resize(image, (500, int(500 * image.shape[0] / image.shape[1])))
        _, buffer = cv2.imencode(".png", resized)
        image_bytes = buffer.tobytes()

        # Draft response per image
        draft = gemini_model.generate_content(
            [doctor_query_template, {"mime_type": "image/png", "data": image_bytes}]
        )

        draft_text = draft.text

        # Extract queries
        query_prompt = f"""
From this draft medical report:

{draft_text}

Generate 3 short PubMed/clinical search queries (≤10 words).
Only return queries, one per line.
"""
        queries_response = gemini_model.generate_content(query_prompt)
        queries = [q.strip() for q in queries_response.text.split("\n") if q.strip()]

        # Run DuckDuckGo
        search_results = "\n".join(
            [f"**{q}:** {search_tool.run(q)}" for q in queries]
        )

        # Refine
        enriched_prompt = f"""
Here is the draft medical report:

{draft_text}

Now refine the "Research References" section by integrating these results:

{search_results}

Return the full updated report for this scan.
"""
        final = gemini_model.generate_content(enriched_prompt)
        results.append((resized, final.text))

    return results


st.set_page_config(page_title="Medical Image Analysis", layout="centered")
st.title("🩺 Medical Image Analysis Tool 🔬")
role = st.radio("Are you a doctor or patient?", ("Patient", "Doctor"))

if role=="Patient":
    st.markdown("""
    Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.), and the AI system will:
    - Analyze the scan
    - Provide structured diagnostic insights
    - Explain findings in simple terms
    - Link recent research
    - Highlight regions of interest with heatmaps
    """)

    uploaded_file = st.sidebar.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.sidebar.button("🔍 Analyze Image"):
            with st.spinner("Analyzing the image... please wait."):
                # Save uploaded image
                image_path = f"temp_image.{uploaded_file.name.split('.')[-1]}"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run analysis
                rgb_img, report = analyze_medical_image(image_path)

                if rgb_img is not None:
                    # Original & Grad-CAM side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(rgb_img, caption="Processed Image", use_container_width=True)
                    with col2:
                        cam_img = generate_gradcam(cv2.imread(image_path))
                        st.image(cam_img, caption="AI Attention Map (Grad-CAM)", use_container_width=True)

                st.subheader("📋 Analysis Report")
                st.markdown(report)

                # Cleanup
                os.remove(image_path)
    else:
        st.info("⚠️ Please upload a medical image to begin analysis.")
elif role == "Doctor":
    uploaded_files = st.sidebar.file_uploader(
        "Upload patient scans", type=["jpg","jpeg","png","bmp"], accept_multiple_files=True
    )
    if uploaded_files and st.sidebar.button("🔍 Analyze All Scans"):
        paths = []
        for f in uploaded_files:
            path = f"temp_{f.name}"
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)

        with st.spinner("Analyzing scans..."):
            reports = analyze_doctor_images(paths)

            for idx, (img, report) in enumerate(reports, 1):
                st.image(img, caption=f"Processed Image {idx}", use_container_width=True)
                cam_img = generate_gradcam(cv2.imread(paths[idx-1]))
                st.image(cam_img, caption=f"Grad-CAM Map {idx}", use_container_width=True)

                st.subheader(f"📋 Report {idx}")
                st.markdown(report)

        # Cleanup
        for p in paths:
            os.remove(p)


