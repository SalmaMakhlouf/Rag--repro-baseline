
# rag-repro-baseline — Repro & benchmark d’un pipeline RAG (FR/EN)

** Pipeline minimal et reproductible pour Retrieval-Augmented Generation (RAG) :
BM25 → dense retriever (SBERT) → fusion RRF → (option) CrossEncoder reranking → évaluation P@k.
Projet personnel de Salma Makhlouf pour candidatures de thèse (NLP/IR, long & structured docs). **  

## 🎯 Pourquoi ce mini-projet ?

Avoir une baseline propre et reproductible pour discuter RAG en contexte de thèse.

Mesurer l’impact des choix de retrieval (BM25 vs dense vs RRF vs reranking CE).

Préparer des ablations (top-k, α de RRF, etc.) et une analyse d’erreurs transférable à des jeux de données plus complexes (documents longs/structurés).

## 🔧 Pipeline
[Corpus .txt] ──► BM25 (lexical)
               └─► SBERT (dense)
                    └─► RRF (fusion, α configurable)
                          └─► (option) CrossEncoder rerank
                                 └─► Éval (Precision@k)

- Dense retriever : sentence-transformers/all-MiniLM-L6-v2
- RRF (Reciprocal Rank Fusion) : α=0.6 par défaut
- Reranking (option) : cross-encoder/ms-marco-MiniLM-L-6-v2
- Éval : Precision@k (macro) + détail par requête

## 📂 Données (mini dev-set fourni)

DATA/Corpus/ – petits documents .txt (FR)
DATA/queries/dev.json – requêtes + gold (doc_id attendus)
⚠️ Pour vos propres jeux de données, gardez 1–3 gold docs par requête et documentez vos critères d’annotation.

## 🚀 Installation & exécution
Option A — GitHub Codespaces (recommandé)

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
pip install rank-bm25 sentence-transformers numpy pyyaml scikit-learn

#### Exécuter le pipeline (attention aux majuscules des dossiers) :
# 1) BM25
python src/index_bm25.py --config Configs/bm25.yaml \
  --queries DATA/queries/dev.json \
  --out Outputs/bm25_candidates.json

# 2) Dense retriever
python src/dense_retriever.py --config Configs/hybrid.yaml \
  --queries DATA/queries/dev.json \
  --out Outputs/dense_candidates.json

# 3) Fusion RRF + (option) CE rerank
python src/rerank_ce.py \
  --bm25 Outputs/bm25_candidates.json \
  --dense Outputs/dense_candidates.json \
  --queries DATA/queries/dev.json \
  --config Configs/hybrid.yaml \
  --out Outputs/hybrid_reranked.json

# 4) Évaluation
python src/eval_patk.py \
  --run Outputs/hybrid_reranked.json \
  --gold DATA/queries/dev.json \
  --out Outputs/metrics.json

## 📏 Métriques

Precision@k (macro-moyenne sur les requêtes) :

$$
\mathrm{P@k}(q)=\frac{\#\{\text{docs pertinents dans les }k\text{ premiers}\}}{k}
\qquad
\mathrm{Macro}\text{-}\mathrm{P@k}
=\frac{1}{|Q|}\sum_{q\in Q}\mathrm{P@k}(q)
$$

Où \(Q\) est l’ensemble des requêtes; `k` est la coupure (par défaut \(k=10\)).

## ✅ Résultats (mini dev-set fourni)

Exemple obtenu sur le dev-set inclus (3 requêtes, 3 docs gold) :
Macro P@10 ≈ 1.00 (attendu car corpus jouet et gold simple)

Fichiers générés :
Outputs/bm25_candidates.json
Outputs/dense_candidates.json
Outputs/hybrid_reranked.json
Outputs/metrics.json ← score final

Vos valeurs réelles dépendront du corpus, des requêtes et de la configuration.

## 🧪 Ablations à tester rapidement

Éditez Configs/hybrid.yaml :
bm25_topk, dense_topk, hybrid_topk
alpha (poids de la RRF)
(option) ajoutez des paramètres chunk_size / overlap si vous chaînez ensuite la Génération.
## Template de suivi (EXPERIMENTS.md)

| Exp | Config (k / α / CE?)                                   | **Macro P@10** | Notes              |
|:---:|:-------------------------------------------------------|:--------------:|--------------------|
| v1  | bm25=50, dense=200, α=0.6,<br>CE=off                   | 0.XX           | baseline RRF       |
| v2  | bm25=100, dense=200, α=0.6,<br>CE=on                   | 0.XX           | +CrossEncoder      |
| v3  | bm25=50, dense=200, α=0.3,<br>CE=on                    | 0.XX           | RRF moins agressif |


## 🔎 Analyse d’erreurs (exemple de structure)

Q2 – “médicament contre douleurs abdominales intestin”
Gold : doc2.txt (Colospa)
Top-1 : doc3.txt (RAG/LLM)
Hypothèse : similarité lexicale faible, manque de synonymes (“antispasmodique”) → améliorer vocabulaire, augmenter dense_topk, tester un modèle FR.

Q3 – “qu est ce que RAG avec LLM et reranking”
Gold : doc3.txt
Observé : correct en top-k, CE renforce la position
Note : CE utile pour formuler “query, passage” plus sémantique.

## 🗂️ Structure du dépôt
Configs/
  bm25.yaml         # corpus_dir, topk
  hybrid.yaml       # dense model, RRF (alpha), topk, CE
DATA/
  Corpus/           # .txt
  queries/
    dev.json        # queries + gold (doc_id)
Outputs/            # résultats générés (.json)
src/
  index_bm25.py     # BM25
  dense_retriever.py# SBERT
  rerank_ce.py      # RRF + (option) CrossEncoder
  eval_patk.py      # Precision@k (macro + détails)
Main.py             # orchestrateur (peut être adapté)
requirements.txt
README.md

## 📌 Points “recherche” mis en avant
Baseline reproductible (scripts + configs)
RRF et CrossEncoder reranking séparés pour ablations propres
P@k macro + notes d’erreurs pour guider l’amélioration
Corpus/queries FR (pertinent pour RAG long/structuré en contexte francophone)

## Pour mettre à jour la section Résultats

Après exécution :
cat Outputs/metrics.json
