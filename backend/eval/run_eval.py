"""
Evaluation harness: runs a set of known question/answer pairs through the
REAL RAG pipeline (real embeddings, real LLM calls) and automatically
scores whether each answer is correct -- this is the automated version
of all the manual testing we did by hand throughout this project.

Two things get checked per case:
1. If expect_refusal is True: does the answer correctly say "couldn't
   find this" instead of guessing?
2. If expect_refusal is False: does the answer contain every expected
   keyword? (A crude check, but effective -- if we expect "8.63" to
   appear and it doesn't, something is badly wrong.)

This uses real API calls, so it costs a small amount of quota/money and
takes longer than the unit tests. Run it after any change to chunking,
retrieval, or prompts -- basically, whenever you want to make sure you
haven't quietly broken grounding accuracy.

Usage:
    python -m backend.eval.run_eval
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.rag.embeddings import get_embedder
from backend.rag.llm import get_llm
from backend.rag.vector_store import VectorStore

load_dotenv()

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./backend/data/vector_store")
RELEVANCE_THRESHOLD = 0.3
TOP_K = 3

REFUSAL_PHRASES = ["couldn't find", "could not find", "don't have", "no information", "not mentioned"]

EVAL_CASES_PATH = Path(__file__).parent / "eval_cases.json"


def is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def run_case(case: dict, store: VectorStore, llm) -> dict:
    question = case["question"]

    matches = store.query(question, top_k=TOP_K)
    relevant = [m for m in matches if m["similarity"] >= RELEVANCE_THRESHOLD]
    answer = llm.generate(question, relevant)

    if case["expect_refusal"]:
        passed = is_refusal(answer)
        reason = "" if passed else "Expected a refusal, but got a confident-looking answer instead."
    else:
        missing = [kw for kw in case["expect_keywords"] if kw.lower() not in answer.lower()]
        passed = len(missing) == 0
        reason = f"Missing expected keyword(s): {missing}" if missing else ""

    return {"question": question, "answer": answer, "passed": passed, "reason": reason}


def main():
    if not EVAL_CASES_PATH.exists():
        print(f"No eval cases found at {EVAL_CASES_PATH}")
        return

    cases = json.loads(EVAL_CASES_PATH.read_text())["cases"]

    embedder = get_embedder()
    llm = get_llm()
    store = VectorStore(embedder=embedder, persist_dir=PERSIST_DIR)

    if store.count() == 0:
        print("Vector store is empty -- run `python -m backend.rag.ingest` first.")
        return

    print(f"Running {len(cases)} eval cases against {store.count()} loaded chunks...\n")

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['question']}")
        result = run_case(case, store, llm)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"  -> {status}")
        if not result["passed"]:
            print(f"     Reason: {result['reason']}")
            print(f"     Answer given: {result['answer'][:200]}")
        print()

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print("=" * 50)
    print(f"RESULT: {passed_count}/{total} passed ({passed_count / total * 100:.0f}%)")

    if passed_count < total:
        print("\nFailed cases:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['question']}")


if __name__ == "__main__":
    main()
