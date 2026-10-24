
from pathlib import Path
from dotenv import load_dotenv
import os

print("Test started", flush=True)

env_file = Path(__file__).with_name(".env")
print("Looking for .env:", env_file.exists(), flush=True)

load_dotenv(dotenv_path=env_file, override=True)

key = os.getenv("OPENAI_API_KEY")

if not key:
    print("ERROR: OPENAI_API_KEY was not found.", flush=True)
else:
    print("API key loaded:", True, flush=True)
    print("Key length:", len(key), flush=True)
    print("Key prefix:", key[:3], flush=True)

print("Test completed", flush=True)
