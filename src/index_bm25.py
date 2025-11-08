import os, argparse, glob, re, numpy as np, json
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def load_corpus(corpus_dir):
    docs, ids = [], []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        ids.append(os.path.basename(path))
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append(f.read())
    return ids, docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bm25.yaml")
    ap.add_argument("--queries", default="data/queries/dev.json")
    ap.add_argument("--out", default="outputs/bm25_candidates.json")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ids, docs = load_corpus(cfg["index"]["corpus_dir"])
    tokenized_corpus = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    qj = json.load(open(args.queries, "r", encoding="utf-8"))
    out = []
    for q in qj["queries"]:
        toks = tokenize(q["text"])
        scores = bm25.get_scores(toks)
        topk = cfg["search"]["topk"]
        idx = np.argsort(scores)[::-1][:topk]
        cand = [{"doc_id": ids[i], "score": float(scores[i])} for i in idx]
        out.append({"qid": q["id"], "query": q["text"], "candidates": cand})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
