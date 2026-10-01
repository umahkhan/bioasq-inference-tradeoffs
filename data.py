from datasets import load_dataset
from common import DATASET, SEED


def load_subset(n_questions, n_docs, split="dev", qtype="summary", min_words=30):
    """First n_questions questions of one type with long gold answers, plus a corpus of all their relevant
    passages padded with random distractors to n_docs."""
    questions = load_dataset(DATASET, "question-answer-passages", split=split)
    questions = questions.filter(lambda q: q["type"] == qtype and len(q["answer"].split()) >= min_words).select(range(n_questions))
    corpus = load_dataset(DATASET, "text-corpus", split="train").shuffle(seed=SEED)
    relevant_ids = {p for q in questions for p in q["relevant_passage_ids"]}
    relevant = list(corpus.filter(lambda d: d["id"] in relevant_ids))
    others = list(corpus.filter(lambda d: d["id"] not in relevant_ids).select(range(n_docs - len(relevant))))
    return list(questions), relevant + others
