import faiss
from sentence_transformers import SentenceTransformer
from common import EMBED_MODEL, require_device

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for bge models


def load_embedder(device):
    return SentenceTransformer(EMBED_MODEL, device=require_device(device))


def build_index(docs, embedder):
    vecs = embedder.encode([f"{d['title']}\n{d['text']}" for d in docs], batch_size=64, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])  # exact cosine search (vectors are normalized)
    index.add(vecs)
    return index


def retrieve(question, index, docs, embedder, k):
    query = embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True)
    _, ids = index.search(query, k)
    return [docs[i] for i in ids[0]]
