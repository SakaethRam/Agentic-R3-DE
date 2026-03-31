# ================================
# SEGMENT 2: COGNITIVE PIPELINE
# ================================

from textblob import TextBlob
from sklearn.cluster import KMeans
import pandas as pd
import networkx as nx

def build_primitives(docs):
    primitives = []

    for speaker, text in docs:
        spacy_doc = nlp(text)

        entities = [(e.text.strip(), e.label_) for e in spacy_doc.ents]

        actions, states = [], []
        for token in spacy_doc:
            if token.pos_ == "VERB" and not token.is_stop:
                subj = [c.text for c in token.children if c.dep_ == "nsubj"]
                obj  = [c.text for c in token.children if c.dep_ in ("dobj", "attr")]
                adv  = [c.text for c in token.children if c.dep_ == "advmod"]

                actions.append(token.lemma_)
                phrase = " ".join(subj + [token.lemma_] + adv + obj)
                if phrase.strip():
                    states.append(phrase)

        primitives.append({
            "speaker": speaker,
            "text": text,
            "entities": entities,
            "actions": list(set(actions)),
            "states": list(set(states)),
            "intent": "question" if "?" in text else "command" if "!" in text else "statement",
            "sentiment": round(TextBlob(text).sentiment.polarity, 2),
        })

    return primitives


def semantic_structuring(primitives, word_vectors):
    embeddings = []

    for p in primitives:
        words = p["actions"] + p["states"]
        valid = [w for w in words if w in word_vectors]

        emb = np.mean(
            [word_vectors.get_vector(w) for w in valid],
            axis=0
        ) if valid else np.zeros(word_vectors.vector_size)

        embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Clustering
    n_clusters = max(2, min(6, len(embeddings))) if len(embeddings) > 1 else 1
    labels = np.zeros(len(embeddings), dtype=int)

    if len(embeddings) > 1:
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embeddings)

    structured = []
    for i, p in enumerate(primitives):
        structured.append({
            "speaker": p["speaker"],
            "entity": p["entities"][0][0] if p["entities"] else "unknown",
            "state": ", ".join(p["states"]),
            "trigger": ", ".join(p["actions"]),
            "context": f"cluster_{labels[i]}",
            "intent": p["intent"],
            "sentiment": p["sentiment"],
            "original_text": p["text"]
        })

    return structured, embeddings, labels, n_clusters


def build_causality(structured):
    G = nx.DiGraph()

    for i in range(len(structured) - 1):
        G.add_edge(
            i, i + 1,
            relation="sequence",
            speaker_from=structured[i]["speaker"],
            speaker_to=structured[i + 1]["speaker"]
        )

    return nx.node_link_data(G)