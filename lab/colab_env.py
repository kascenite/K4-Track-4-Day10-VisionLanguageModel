"""Đặt biến môi trường Kaggle từ Colab Secrets (không ghi khoá vào notebook)."""
import os

from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
if os.environ["KAGGLE_KEY"].startswith("KGAT_"):  # token API kiểu mới của Kaggle
    os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_KEY"]
print("kaggle user:", os.environ["KAGGLE_USERNAME"])
