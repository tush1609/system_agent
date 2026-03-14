from dotenv import load_dotenv
from pathlib import Path
import os
import yaml

class AppConfig:
    def __init__(self, path="config.yaml"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, path)

        with open(full_path) as f:
            data = yaml.safe_load(f)

        env_path = Path('.env')
        load_dotenv(env_path)   # ✅ loads .env into os.environ

        self.gpt_key = os.getenv('OPENAI_API_KEY')
        self.hugging_face_key = os.getenv('HUGGING_FACE_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')

# Singleton — import this in all files
config = AppConfig()
