from langchain_google_genai import ChatGoogleGenerativeAI
from configuration.config_loader import config
from .models import Models

class Gemini(Models):
    def __init__(self):
        self.gemini_model = ''

    def gemini_2_flash(self):
        self.gemini_model = 'gemini-2.0-flash'
        return self

    def gemini_2_flash_lite(self):
        self.gemini_model = 'gemini-2.0-flash-lite'
        return self

    def gemini_2_pro(self):
        self.gemini_model = 'gemini-2.5-pro'
        return self

    def gemini_1_5_flash(self):
        self.gemini_model = 'gemini-1.5-flash'
        return self

    def gemini_1_5_pro(self):
        self.gemini_model = 'gemini-1.5-pro'
        return self

    def gemini_3_1_flash_lite_preview(self):
        self.gemini_model = 'gemini-3.1-flash-lite-preview'
        return self

    def model(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.gemini_model,
            google_api_key=config.gemini_key,
            temperature=0,
            max_tokens=20
        )
