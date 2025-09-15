import cv2
import numpy as np
from config import gemini_model, patient_query_template
from utils.image_utils import preprocess_image
from utils.search_utils import extract_queries, run_search

def analyze_medical_image(image_path: str):
    try:
        # Preprocess
        resized, rgb_image = preprocess_image(image_path)

        # Convert image to bytes
        _, buffer = cv2.imencode(".png", resized)
        image_bytes = buffer.tobytes()

        # Step 1: Draft report
        draft_response = gemini_model.generate_content(
            [patient_query_template, {"mime_type": "image/png", "data": image_bytes}]
        )
        draft_text = draft_response.text

        # Step 2: Extract queries
        queries = extract_queries(draft_text)

        # Step 3: Search
        search_results = run_search(queries)

        # Step 4: Refine report
        enriched_prompt = f"""
Here is the draft medical report:

{draft_text}

Now refine ONLY the "Research Context" section using:

{search_results}

Return the full updated report.
"""
        final_response = gemini_model.generate_content(enriched_prompt)
        return rgb_image, final_response.text

    except Exception as e:
        return None, f"⚠️ Analysis error: {e}"
