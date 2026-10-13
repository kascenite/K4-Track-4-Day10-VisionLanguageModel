"""Đặt biến môi trường Kaggle từ Colab Secrets (không ghi khoá vào notebook)."""
import os

from google.colab import userdata

key = userdata.get("KAGGLE_KEY")
if key.startswith("KGAT_"):  # token API kiểu mới của Kaggle: không cần username
    os.environ["KAGGLE_API_TOKEN"] = key
else:
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = key
print("kaggle credentials set")
