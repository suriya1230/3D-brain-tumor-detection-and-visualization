"""Phase 3 knowledge/retrieval subsystem."""

from dotenv import load_dotenv

# Loads backend/.env (searches upward from cwd) so GROQ_API_KEY,
# ANTHROPIC_API_KEY, NCBI_API_KEY etc. don't need manual shell exports.
# A no-op if no .env file exists — nothing here requires one.
load_dotenv()
