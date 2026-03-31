# ================================
# SEGMENT 3: UNCERTAINTY + OUTPUT
# ================================

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np

def uncertainty_layer(embeddings, labels):
    std = np.full(len(embeddings), 0.35)

    if len(embeddings) > 1:
        try:
            kernel = ConstantKernel(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel)
            gp.fit(embeddings, labels)
            _, std = gp.predict(embeddings, return_std=True)
        except:
            pass

    return std


def enrich_data(structured, std):
    enriched = []

    for i, item in enumerate(structured):
        conf = 1 - (std[i] / max(std.max(), 1e-6))
        conf = min(0.98, max(0.1, conf))

        item["confidence"] = round(conf, 3)
        item["noise_level"] = "high" if std[i] > 0.7 else "medium" if std[i] > 0.4 else "low"
        item["bias"] = "positive" if item["sentiment"] > 0.2 else "negative" if item["sentiment"] < -0.2 else "neutral"

        enriched.append(item)

    return enriched


async def main():
    async with Actor:
        input_data = await Actor.get_input() or {}

        url = input_data.get("url")
        raw_text = input_data.get("rawText")

        # Input Handling
        if raw_text:
            doc = nlp(raw_text)
            docs = [("unknown", s.text.strip()) for s in doc.sents if s.text.strip()]
        else:
            sentences = await scrape_raw_data(url)
            docs = [("unknown", s) for s in sentences]

        # Pipeline
        primitives = build_primitives(docs)
        structured, embeddings, labels, n_clusters = semantic_structuring(primitives, word_vectors)
        graph = build_causality(structured)

        std = uncertainty_layer(embeddings, labels)
        enriched = enrich_data(structured, std)

        # Output
        for record in enriched:
            await Actor.push_data(record)

        await Actor.set_value("GRAPH", graph)

        Actor.log.info("Prototype pipeline complete.")