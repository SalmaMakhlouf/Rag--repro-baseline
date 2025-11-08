# rag-repro-baseline
**Reproduction & benchmark d'un pipeline RAG : BM25 → dense retriever → reranking (CrossEncoder) + P@k & ablations.**

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Exécution
```bash
python Main.py
# sorties: outputs/*candidates.json, outputs/hybrid_reranked.json, outputs/metrics.json
```

## Données
- Mini corpus d'exemple fourni dans `data/corpus/`
- Requêtes + gold: `data/queries/dev.json`

## Ablations à tester
- `configs/hybrid.yaml` : alpha, *_topk
- (optionnel) chunk_size, ctx_length si vous ajoutez une étape génération

_2025-11-08_
