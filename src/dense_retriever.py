import os, argparse, glob, json, numpy as np
from sentence_transformers import SentenceTransformer, util

def load_corpus(corpus_dir):
    docs, ids = [], []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        ids.append(os.path.basename(path))
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append(f.read().strip())
    return ids, docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hybrid.yaml")
    ap.add_argument("--queries", default="data/queries/dev.json")
    ap.add_argument("--out", default="outputs/dense_candidates.json")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model = SentenceTransformer(cfg["index"]["dense_model"])

    ids, docs = load_corpus(cfg["index"]["corpus_dir"])
    corpus_emb = model.encode(docs, convert_to_tensor=True, normalize_embeddings=True)

    qj = json.load(open(args.queries, "r", encoding="utf-8"))
    out = []
    for q in qj["queries"]:
        q_emb = model.encode([q["text"]], convert_to_tensor=True, normalize_embeddings=True)[0]
        scores = util.cos_sim(q_emb, corpus_emb).cpu().numpy().ravel()
        idx = np.argsort(scores)[::-1][:cfg["search"]["dense_topk"]]
        cand = [{"doc_id": ids[i], "score": float(scores[i])} for i in idx]
        out.append({"qid": q["id"], "query": q["text"], "candidates": cand})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
