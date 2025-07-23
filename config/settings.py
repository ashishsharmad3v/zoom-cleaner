import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
DEFAULT_MODEL = "gpt-3.5-turbo"
ADVANCED_MODEL = "gpt-4"

# Processing Configuration
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 500
MAX_WORKERS = 4
CONTEXT_WINDOW = 1000

# Quality Settings
QUALITY_THRESHOLD = 80
MAX_RETRIES = 3