from dotenv import load_dotenv
from pathlib import Path
import os
import yaml


class AppConfig:
    def __init__(self, path="config.yaml"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, path)

        with open(full_path) as f:
            data = yaml.safe_load(f) or {}

        env_path = Path('.env')
        load_dotenv(env_path)   # ✅ loads .env into os.environ

        self.gpt_key = os.getenv('OPENAI_API_KEY')
        self.hugging_face_key = os.getenv('HUGGING_FACE_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.debug = data.get("debug", False)

        logger_config = data.get("logger", {})
        self.logger_info_enabled = logger_config.get("info_enabled", True)
        self.logger_debug_enabled = logger_config.get("debug_enabled", True)
        self.logger_error_enabled = logger_config.get("error_enabled", True)
        self.logger_warn_enabled = logger_config.get("warn_enabled", True)
        self.third_party_logs_enabled = logger_config.get("third_party_logs_enabled", True)
        self.file_logs_enabled = logger_config.get("file_logs_enabled", True)
        self.terminal_logs_enabled = logger_config.get("terminal_logs_enabled", True)


# Singleton — import this in all files
config = AppConfig()
