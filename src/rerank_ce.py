import argparse, json, os
from collections import defaultdict

def reciprocal_rank_fusion(*runs, k=50, alpha=0.6):
    fused = {}
    for qid in runs[0].keys():
        scores = defaultdict(float)
        for run in runs:
            for rank, (doc, _) in enumerate(run[qid][:k], start=1):
                scores[doc] += 1.0 / (alpha + rank)
        fused[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused

def load_run(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    run = {}
    for item in data:
        run[item["qid"]] = [(c["doc_id"], c["score"]) for c in item["candidates"]]
    return run

def to_list(fused, topk):
    out = []
    for qid, pairs in fused.items():
        out.append({"qid": qid, "candidates": [{"doc_id": d, "score": float(s)} for d,s in pairs[:topk]]})
    return out

def cross_encoder_rerank(candidates_json, query_json, model_name, topk):
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        print("CrossEncoder unavailable; skipping rerank.")
        return candidates_json
    qmap = {q["id"]: q["text"] for q in query_json["queries"]}
    ce = CrossEncoder(model_name)
    out = []
    for item in candidates_json:
        qid = item["qid"]
        pairs = [(qmap[qid], c["doc_id"]) for c in item["candidates"]]
        scores = ce.predict(pairs)
        ranked = sorted(zip(item["candidates"], scores), key=lambda x: x[1], reverse=True)[:topk]
        out.append({"qid": qid, "candidates": [{"doc_id": c["doc_id"], "score": float(s)} for c,s in ranked]})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", default="outputs/bm25_candidates.json")
    ap.add_argument("--dense", default="outputs/dense_candidates.json")
    ap.add_argument("--queries", default="data/queries/dev.json")
    ap.add_argument("--config", default="configs/hybrid.yaml")
    ap.add_argument("--out", default="outputs/hybrid_reranked.json")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    bm25 = load_run(args.bm25)
    dense = load_run(args.dense)
    fused = reciprocal_rank_fusion(bm25, dense, k=cfg["search"]["bm25_topk"], alpha=cfg["search"]["alpha"])
    fused_list = to_list(fused, cfg["search"]["hybrid_topk"])

    qjson = json.load(open(args.queries, "r", encoding="utf-8"))
    final = cross_encoder_rerank(fused_list, qjson, cfg["rerank"]["model"], cfg["rerank"]["topk"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(final, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
