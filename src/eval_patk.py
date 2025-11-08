import argparse, json

def precision_at_k(run_path, gold_path, k=10):
    run = json.load(open(run_path, "r", encoding="utf-8"))
    qjson = json.load(open(gold_path, "r", encoding="utf-8"))
    gold = qjson["gold"]
    total=0; details=[]
    for item in run:
        preds = [c["doc_id"] for c in item["candidates"][:k]]
        gset = set(gold.get(item["qid"], []))
        hits = sum(1 for p in preds if p in gset)
        prec = hits / max(len(preds),1)
        details.append({"qid": item["qid"], "p@k": prec, "hits": hits, "preds": preds, "gold": list(gset)})
        total += prec
    macro = total / max(len(run),1)
    return {"macro_p@k": macro, "details": details}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/hybrid_reranked.json")
    ap.add_argument("--gold", default="data/queries/dev.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="outputs/metrics.json")
    args = ap.parse_args()
    metrics = precision_at_k(args.run, args.gold, args.k)
    json.dump(metrics, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
