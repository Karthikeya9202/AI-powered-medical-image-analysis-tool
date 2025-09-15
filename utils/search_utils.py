from config import gemini_model, search_tool

def extract_queries(draft_text, max_queries=3):
    """Generate short search queries from draft report using Gemini."""
    query_prompt = f"""
From this draft medical report:

{draft_text}

Generate {max_queries} short search queries (≤10 words).
Only return queries, one per line.
"""
    response = gemini_model.generate_content(query_prompt)
    return [q.strip() for q in response.text.split("\n") if q.strip()]

def run_search(queries):
    """Run DuckDuckGo searches and return formatted results."""
    return "\n".join([f"**{q}:** {search_tool.run(q)}" for q in queries])
