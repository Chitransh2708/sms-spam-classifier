import nltk
import os

NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)

nltk.data.path.append(NLTK_DIR)

nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)
