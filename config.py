import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat"
    TEMPERATURE = 0

    @classmethod
    def validate(cls):
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("错误: 请在 .env 文件中设置您的 DEEPSEEK_API_KEY")