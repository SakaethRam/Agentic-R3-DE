# ================================
# SEGMENT 1: INPUT + DATA LAYER
# ================================

import asyncio
import os
import sys
import re
from apify import Actor
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import spacy
import gensim.downloader as api
import numpy as np

# ─── Load spaCy ─────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    Actor.log.info("Installing spaCy model...")
    os.system(f"{sys.executable} -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
    nlp = spacy.load("en_core_web_sm")

# ─── Load Embeddings ────────────
try:
    word_vectors = api.load("glove-wiki-gigaword-100")
except Exception:
    class ZeroVec:
        vector_size = 100
        def __contains__(self, item): return False
        def get_vector(self, word): return np.zeros(100)
    word_vectors = ZeroVec()

# ─── Scraper ────────────────────
async def scrape_raw_data(url: str) -> list[str]:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return [s.strip() for s in text.split(".") if s.strip()]

    except Exception as e:
        Actor.log.error(f"Scrape error: {e}")
        return []