# lmms_eval/DDCoT/prompts.py

SPLIT_PROMPT = """You decompose a visual question into necessary sub-questions.
Return ONLY valid JSON:
{"sub_questions":["...","..."]}

Rules:
- 3~10 sub-questions
- Do NOT answer them here
"""

NEGSPACE_PROMPT = """Answer sub-questions but assume you CANNOT see the image.
Return ONLY valid JSON:
{"answers":[{"q":"...","a":"..."}, ...]}

Rules:
- If it requires image content (objects/text/colors/positions/counts), output exactly "Uncertain".
"""

VQA_PROMPT = """Answer based on the image. Be concise and factual.
Question: {q}
"""

JOINT_PROMPT = """Solve the original question using the evidence.
Evidence may contain noise; ignore inconsistent items.

Return ONLY valid JSON:
{"final_answer":"...","rationale":"..."}
Rules:
- final_answer should be concise
- rationale 1~4 sentences
"""
