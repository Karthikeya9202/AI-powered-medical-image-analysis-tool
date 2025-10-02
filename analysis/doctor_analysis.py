import cv2
from config import gemini_model, doctor_query_template
from utils.image_utils import preprocess_image
from utils.search_utils import extract_queries, run_search

def analyze_doctor_images(image_paths: list):
    results = []

    for path in image_paths:
        # Preprocess
        resized, _ = preprocess_image(path)

        # Convert to bytes for Gemini
        _, buffer = cv2.imencode(".png", resized)
        image_bytes = buffer.tobytes()

        # Draft report
        draft = gemini_model.generate_content(
            [doctor_query_template, {"mime_type": "image/png", "data": image_bytes}]
        )
        draft_text = draft.text

        # Extract queries for references
        queries = extract_queries(draft_text)
        search_results = run_search(queries)

        # Refine report
        enriched_prompt = f"""
Here is the draft medical report:

{draft_text}

Now refine the "Research References" section using:

{search_results}

Return the full updated report for this scan.
"""
        final = gemini_model.generate_content([enriched_prompt])
        results.append((resized, final.text))  # Use preprocessed image

    return results
