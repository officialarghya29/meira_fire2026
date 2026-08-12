"""
datasets_fire.py
================
Two realistic IR benchmark datasets aligned with FIRE / ACM SIGIR conventions.

Dataset 1: FIRE-AgentIR-2026
  - Multi-turn agentic conversation retrieval (Topics 3+6 of FIRE CFP)
  - 10 IR/NLP topic clusters, 6 turns/conv, hard same-domain negatives
  - Metric focus: nDCG@10, MAP, MRR

Dataset 2: FIRE-CrossLingIR-2026
  - Cross-lingual / Indian-language information retrieval
  - Simulates Hindi-English, Bengali-English bilingual relevance judgements
  - Vocabulary shift, transliteration noise, code-switching
  - Aligned with FIRE's historical emphasis on Indian language IR

Both datasets expose a unified DatasetSplit API compatible with
k-fold cross-validation and multi-seed experiments.
"""

import random, math
import numpy as np
from typing import List, Dict, Tuple, Optional

# ── Shared IR vocabulary ───────────────────────────────────────────────
BASE_IR = ["retrieval","ranking","query","document","relevance","score",
           "model","neural","embedding","index","search","evaluation",
           "metric","user","system","learning","representation",
           "transformer","attention","context","passage","token","bert",
           "dense","sparse","recall","precision","ndcg","map","mrr"]

# ══════════════════════════════════════════════════════════════════════
# DATASET 1: FIRE-AgentIR-2026
# ══════════════════════════════════════════════════════════════════════

AGENT_TOPICS = {
    "episodic_memory_ir": {
        "core":  ["episodic","memory","retrieval","agent","temporal","context",
                  "history","decay","slot","write","forgetting","consolidation"],
        "hard_neg_sibling": "dense_retrieval_methods",
    },
    "dense_retrieval_methods": {
        "core":  ["dense","bi-encoder","faiss","contrastive","knn","vector",
                  "approximate","nearest","passage","dpr","colbert","ance"],
        "hard_neg_sibling": "episodic_memory_ir",
    },
    "rag_and_generation": {
        "core":  ["generation","augmented","rag","grounding","knowledge","prompt",
                  "hallucination","faithfulness","extractive","abstractive"],
        "hard_neg_sibling": "neural_reranking",
    },
    "neural_reranking": {
        "core":  ["reranking","pointwise","pairwise","listwise","monobert",
                  "cross-encoder","fine-tuning","relevance","classification"],
        "hard_neg_sibling": "rag_and_generation",
    },
    "explainability_ir": {
        "core":  ["explainability","attribution","saliency","xai","fairness",
                  "bias","transparency","rationale","lime","shap"],
        "hard_neg_sibling": "evaluation_metrics",
    },
    "evaluation_metrics": {
        "core":  ["ndcg","map","mrr","precision","recall","bpref","rbp",
                  "qrels","relevance","judgment","annotation","pooling"],
        "hard_neg_sibling": "explainability_ir",
    },
    "conversational_search": {
        "core":  ["conversational","dialogue","clarification","session","turn",
                  "reformulation","resolution","coreference","mixed-initiative"],
        "hard_neg_sibling": "query_understanding",
    },
    "query_understanding": {
        "core":  ["query","intent","expansion","reformulation","ambiguity",
                  "suggestion","completion","segmentation","entity","tagging"],
        "hard_neg_sibling": "conversational_search",
    },
    "multimodal_ir": {
        "core":  ["multimodal","image","visual","cross-modal","fusion","caption",
                  "clip","vit","grounding","alignment","retrieval"],
        "hard_neg_sibling": "dense_retrieval_methods",
    },
    "agentic_systems_ir": {
        "core":  ["agentic","tool","planning","react","chain","reasoning",
                  "workflow","memory","action","observation","reflection"],
        "hard_neg_sibling": "episodic_memory_ir",
    },
}
AGENT_TOPIC_NAMES = list(AGENT_TOPICS.keys())

# ══════════════════════════════════════════════════════════════════════
# DATASET 2: FIRE-CrossLingIR-2026
# ══════════════════════════════════════════════════════════════════════

CROSSLING_TOPICS = {
    "hindi_english_health": {
        "core_en":  ["health","disease","medicine","treatment","symptom",
                     "hospital","patient","diagnosis","drug","clinical"],
        "core_hi":  ["swasthya","bimari","dawa","ilaj","hospital","rogi",
                     "doctor","aushadhi","chikitsa","lakshan"],
        "hard_neg_sibling": "hindi_english_agriculture",
    },
    "hindi_english_agriculture": {
        "core_en":  ["agriculture","crop","farmer","irrigation","soil","yield",
                     "fertilizer","harvest","seed","monsoon"],
        "core_hi":  ["kheti","kisan","fasal","sinchai","mitti","upaj",
                     "khad","katai","beej","barsaat"],
        "hard_neg_sibling": "hindi_english_health",
    },
    "bengali_english_news": {
        "core_en":  ["news","politics","election","government","parliament",
                     "policy","minister","vote","party","economy"],
        "core_hi":  ["sambad","rajneeti","nirbachon","sarkar","sansad",
                     "niti","mantri","mat","dal","arthaneeti"],
        "hard_neg_sibling": "bengali_english_education",
    },
    "bengali_english_education": {
        "core_en":  ["education","school","student","teacher","curriculum",
                     "examination","university","learning","literacy","admission"],
        "core_hi":  ["shiksha","vidyalaya","chatra","shikshak","pathyakram",
                     "pariksha","vishwavidyalaya","siksha","saksharata","pravesh"],
        "hard_neg_sibling": "bengali_english_news",
    },
    "tamil_english_technology": {
        "core_en":  ["technology","software","internet","computer","digital",
                     "startup","innovation","engineering","mobile","data"],
        "core_hi":  ["thozhilnuNpam","maenporuL","inaiyal","kaNiNi","digital",
                     "tokuthi","padaippu","payaniyal","kaippesi","thagaval"],
        "hard_neg_sibling": "hindi_english_agriculture",
    },
}
CROSSLING_TOPIC_NAMES = list(CROSSLING_TOPICS.keys())


# ── Tokeniser (shared fixed vocab → stable across datasets) ───────────
VOCAB_SIZE = 30522
_WORD2ID: Dict[str,int] = {}

def word2id(w: str) -> int:
    if w not in _WORD2ID:
        _WORD2ID[w] = (len(_WORD2ID) % (VOCAB_SIZE - 4)) + 4
    return _WORD2ID[w]

def tokenise(text: str, max_len: int = 128) -> Tuple[List[int], List[int]]:
    tokens = [1]  # CLS
    for w in text.lower().split()[:max_len - 2]:
        tokens.append(word2id(w))
    tokens.append(2)  # SEP
    ids  = tokens[:max_len]
    mask = [1] * len(ids)
    pad  = max_len - len(ids)
    return ids + [0]*pad, mask + [0]*pad


# ── Text builder helpers ───────────────────────────────────────────────

def _build_agent_text(topic_name: str, is_pos: bool, turn: int,
                       rng: random.Random, noise: float = 0.0) -> str:
    t = AGENT_TOPICS[topic_name]
    core = t["core"]
    if is_pos:
        drop = min(turn * 0.07, 0.35)          # harder later turns
        n_core = max(2, int(len(core) * (1 - drop)))
        words  = (rng.sample(core, n_core) +
                  rng.sample(BASE_IR, rng.randint(2, 4)))
    else:
        # hard negative: sibling topic + shared IR terms
        sib_name = t["hard_neg_sibling"]
        sib_core = AGENT_TOPICS[sib_name]["core"]
        words    = (rng.sample(sib_core, rng.randint(2, 4)) +
                    rng.sample(BASE_IR, rng.randint(3, 5)))
    if noise > 0:
        words += rng.sample(BASE_IR, int(len(words) * noise))
    rng.shuffle(words)
    return " ".join(words)


def _build_crossling_text(topic_name: str, is_pos: bool, lang: str,
                            rng: random.Random) -> str:
    t = CROSSLING_TOPICS[topic_name]
    en_words = t["core_en"]
    hi_words = t["core_hi"]

    if is_pos:
        if lang == "en":
            words = rng.sample(en_words, rng.randint(4, 7)) + rng.sample(BASE_IR, 2)
        elif lang == "hi":
            words = rng.sample(hi_words, rng.randint(4, 7)) + rng.sample(en_words, 1)
        else:  # mixed code-switching
            words = (rng.sample(en_words, 3) + rng.sample(hi_words, 3) +
                     rng.sample(BASE_IR, 2))
    else:
        sib_name = t["hard_neg_sibling"]
        sib = CROSSLING_TOPICS[sib_name]
        words = (rng.sample(sib["core_en"], 3) +
                 rng.sample(sib.get("core_hi", sib["core_en"]), 2) +
                 rng.sample(BASE_IR, 3))
    rng.shuffle(words)
    return " ".join(words)


# ── Unified sample class ───────────────────────────────────────────────

class IRSample:
    __slots__ = ("input_ids","attention_mask","label","conv_id","turn","hard","dataset")
    def __init__(self, text, label, conv_id, turn, hard, dataset, max_len=128):
        ids, mask = tokenise(text, max_len)
        self.input_ids      = ids
        self.attention_mask = mask
        self.label          = label
        self.conv_id        = conv_id
        self.turn           = turn
        self.hard           = hard
        self.dataset        = dataset   # "agent" or "crossling"


# ── Dataset builders ───────────────────────────────────────────────────

def build_agent_ir_dataset(n_convs: int = 400, turns: int = 6,
                            neg_per_pos: int = 3, hard_ratio: float = 0.65,
                            label_noise: float = 0.05,
                            max_len: int = 128,
                            seed: int = 42) -> List[IRSample]:
    rng = random.Random(seed)
    samples = []
    for cid in range(n_convs):
        topic = rng.choice(AGENT_TOPIC_NAMES)
        for turn in range(turns):
            q   = _build_agent_text(topic, True, turn, rng, 0.1)
            doc = _build_agent_text(topic, True, turn, rng, 0.15)
            lbl = 1
            if rng.random() < label_noise: lbl = 1 - lbl
            samples.append(IRSample(q+" [SEP] "+doc, lbl, cid, turn, False, "agent", max_len))

            for _ in range(neg_per_pos):
                is_hard = rng.random() < hard_ratio
                neg_doc = _build_agent_text(topic, False, turn, rng, 0.2)
                lbl = 0
                if rng.random() < label_noise: lbl = 1 - lbl
                samples.append(IRSample(q+" [SEP] "+neg_doc, lbl, cid, turn, is_hard, "agent", max_len))

    rng.shuffle(samples)
    return samples


def build_crossling_ir_dataset(n_convs: int = 200, turns: int = 4,
                                neg_per_pos: int = 3, hard_ratio: float = 0.60,
                                label_noise: float = 0.06,
                                max_len: int = 128,
                                seed: int = 42) -> List[IRSample]:
    rng    = random.Random(seed)
    langs  = ["en","hi","mixed"]
    samples = []
    for cid in range(n_convs):
        topic = rng.choice(CROSSLING_TOPIC_NAMES)
        for turn in range(turns):
            lang = rng.choice(langs)
            q    = _build_crossling_text(topic, True, lang, rng)
            doc  = _build_crossling_text(topic, True, rng.choice(langs), rng)
            lbl  = 1
            if rng.random() < label_noise: lbl = 1 - lbl
            samples.append(IRSample(q+" [SEP] "+doc, lbl, cid, turn, False, "crossling", max_len))

            for _ in range(neg_per_pos):
                is_hard = rng.random() < hard_ratio
                neg_doc = _build_crossling_text(topic, False, rng.choice(langs), rng)
                lbl = 0
                if rng.random() < label_noise: lbl = 1 - lbl
                samples.append(IRSample(q+" [SEP] "+neg_doc, lbl, cid, turn, is_hard, "crossling", max_len))

    rng.shuffle(samples)
    return samples


# ── Split utilities ────────────────────────────────────────────────────

def stratified_split(samples: List[IRSample],
                     ratios: Tuple[float,...] = (0.70, 0.15, 0.15),
                     seed: int = 42) -> Tuple[List[IRSample],...]:
    rng = random.Random(seed)
    pos = [s for s in samples if s.label == 1]
    neg = [s for s in samples if s.label == 0]
    rng.shuffle(pos); rng.shuffle(neg)

    def _cut(lst):
        n = len(lst)
        i1 = int(n * ratios[0])
        i2 = int(n * (ratios[0] + ratios[1]))
        return lst[:i1], lst[i1:i2], lst[i2:]

    tr_p,vl_p,te_p = _cut(pos)
    tr_n,vl_n,te_n = _cut(neg)
    tr = tr_p+tr_n; vl = vl_p+vl_n; te = te_p+te_n
    rng.shuffle(tr); rng.shuffle(vl); rng.shuffle(te)
    return tr, vl, te


def kfold_split(samples: List[IRSample], k: int = 5,
                seed: int = 42) -> List[Tuple[List[IRSample], List[IRSample]]]:
    """Returns list of (train, val) folds."""
    rng = random.Random(seed)
    pos = [s for s in samples if s.label == 1]
    neg = [s for s in samples if s.label == 0]
    rng.shuffle(pos); rng.shuffle(neg)

    def _chunks(lst):
        n   = len(lst)
        sz  = n // k
        return [lst[i*sz:(i+1)*sz] for i in range(k)]

    pos_folds = _chunks(pos)
    neg_folds = _chunks(neg)
    folds = []
    for i in range(k):
        val   = pos_folds[i] + neg_folds[i]
        train = (sum(pos_folds[:i]+pos_folds[i+1:],[]) +
                 sum(neg_folds[:i]+neg_folds[i+1:],[]))
        rng.shuffle(val); rng.shuffle(train)
        folds.append((train, val))
    return folds


def dataset_stats(samples: List[IRSample], name: str = "") -> dict:
    n      = len(samples)
    n_pos  = sum(s.label==1 for s in samples)
    n_hard = sum(s.hard for s in samples)
    return {"name": name, "total": n, "positives": n_pos,
            "negatives": n-n_pos, "hard_neg": n_hard,
            "pos_ratio": round(n_pos/n,3)}
