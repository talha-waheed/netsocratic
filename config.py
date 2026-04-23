import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_CLARIFY_ROUNDS: int = int(os.environ.get("MAX_CLARIFY_ROUNDS", "5"))
NUM_CANDIDATES: int = int(os.environ.get("NUM_CANDIDATES", "3"))
KB_DIR: str = os.environ.get("KB_DIR", "agents/knowledge-base")
TOPO_DIR: str = os.environ.get("TOPO_DIR", "../cs598_LMZ_final/topo")
RESULTS_DIR: str = os.environ.get("RESULTS_DIR", "results")
