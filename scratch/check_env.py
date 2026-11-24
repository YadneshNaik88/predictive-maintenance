"""
scratch/check_env.py - Minimal script to verify Groq credentials & connectivity.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env without overriding active process environment variables
load_dotenv(override=False)

key = os.environ.get("GROQ_API_KEY", "").strip()
model = os.environ.get("GROQ_MODEL", "").strip()

key_status = "set" if key else "not set"
model_status = model if model else "not set"

print(f"GROQ_API_KEY: {key_status}")
print(f"GROQ_MODEL: {model_status}")

if key and model:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        llm = ChatGroq(
            model=model,
            groq_api_key=key,
            temperature=0.0,
            request_timeout=20.0,
            max_retries=2,
        )
        res = llm.invoke([HumanMessage(content="reply with the word OK")])
        print(res.content.strip())
    except Exception as e:
        err_msg = str(e)
        if key in err_msg:
            err_msg = err_msg.replace(key, "[REDACTED]")
        print(f"Error: {err_msg}")
