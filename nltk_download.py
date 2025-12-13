import nltk
import os

NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)

nltk.data.path = [NLTK_DIR]

for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, download_dir=NLTK_DIR)
