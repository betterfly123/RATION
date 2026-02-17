import json, re
from typing import Optional, Tuple

def extract_json(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def normalize_choice(x: str, valid=("A","B","C","D","E","F")) -> Optional[str]:
    if not x:
        return None
    x = x.strip().upper()
    m = re.search(r"\b([A-F])\b", x)
    if not m:
        return None
    c = m.group(1)
    return c if c in valid else None
