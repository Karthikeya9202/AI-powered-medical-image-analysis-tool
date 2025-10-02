import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun

GOOGLE_API_KEY = "AIzaSyC6HjV3CNd3hH9PSJkTxNerb5m9qR-BGRo"
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set GOOGLE_API_KEY as an environment variable or replace YOUR_API_KEY_HERE.")
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Vision Model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# DuckDuckGo Search
search_tool = DuckDuckGoSearchRun()

patient_query_template = """
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
