


cat > eval_system/metrics.py << 'EOF'
import re
import string
from typing import List, Dict, Any, Optional

# Try to use NLTK; fallback to simple whitespace tokenization if not available
try:
    import nltk  # type: ignore
    try:
        from nltk.tokenize import word_tokenize  # type: ignore
        _HAS_NLTK = True
    except Exception:
        _HAS_NLTK = False
except Exception:
    _HAS_NLTK = False

def _simple_tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+|[^\w\s]", text.lower()) if t.strip()]

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    if _HAS_NLTK:
        try:
            return [t.lower() for t in word_tokenize(text)]
        except Exception:
            return _simple_tokenize(text)
    return _simple_tokenize(text)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize(text: str) -> str:
    if text is None:
        return ""
    return text.strip().lower().translate(_PUNCT_TABLE)

def exact_match(prediction: str, reference: str) -> int:
    """Case/punct-insensitive exact string match (returns 1 or 0)."""
    return int(normalize(prediction) == normalize(reference))

def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 using lowercased tokens."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    from collections import Counter
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def rouge_scores(prediction: str, references: List[str]) -> Dict[str, float]:
    """Average ROUGE-1 and ROUGE-L F1 over references, with fallback if rouge-score missing."""
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = []
        for ref in references:
            s = scorer.score(ref, prediction)
            scores.append({"rouge1": s["rouge1"].fmeasure, "rougeL": s["rougeL"].fmeasure})
        if not scores:
            return {"rouge1": 0.0, "rougeL": 0.0}
        return {
            "rouge1": sum(s["rouge1"] for s in scores) / len(scores),
            "rougeL": sum(s["rougeL"] for s in scores) / len(scores),
        }
    except Exception:
        def _unigram_f1(a: str, b: str) -> float:
            return f1_score(a, b)
        def _lcs(a_tokens: List[str], b_tokens: List[str]) -> int:
            m, n = len(a_tokens), len(b_tokens)
            dp = [0] * (n + 1)
            for i in range(1, m + 1):
                prev = 0
                for j in range(1, n + 1):
                    temp = dp[j]
                    if a_tokens[i-1] == b_tokens[j-1]:
                        dp[j] = prev + 1
                    else:
                        dp[j] = max(dp[j], dp[j-1])
                    prev = temp
            return dp[n]
        r1_vals, rl_vals = [], []
        for ref in references:
            r1_vals.append(_unigram_f1(prediction, ref))
            a, b = tokenize(prediction), tokenize(ref)
            if not a or not b:
                rl_vals.append(0.0)
            else:
                lcs = _lcs(a, b)
                prec = lcs / len(a)
                rec = lcs / len(b)
                rl_vals.append(0.0 if (prec + rec) == 0 else 2*prec*rec/(prec+rec))
        if not r1_vals:
            return {"rouge1": 0.0, "rougeL": 0.0}
        return {"rouge1": sum(r1_vals)/len(r1_vals), "rougeL": sum(rl_vals)/len(rl_vals)}

# --- Simple "hallucination / citation" heuristics ---

_CITATION_PATTERNS = [
    re.compile(r"\[(?:doc)?\d+\]"),             # [1], [2], [doc3]
    re.compile(r"\(\s*source\s*:[^)]+\)", re.I),# (Source: ...)
    re.compile(r"https?://\S+"),                # URLs
]

def extract_citations(text: str) -> List[str]:
    cites: List[str] = []
    for pat in _CITATION_PATTERNS:
        cites.extend(m.group(0) for m in pat.finditer(text or ""))
    norm = []
    for c in cites:
        m = re.match(r"\[((?:doc)?\d+)\]", c, flags=re.I)
        if m:
            norm.append(m.group(1).lower())
        else:
            norm.append(c)
    return norm

def hallucination_check(prediction: str, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Heuristic 'citation coverage' check (structure proxy, not factual correctness)."""
    cites = extract_citations(prediction or "")
    sources = set()
    if ctx and isinstance(ctx.get("sources"), list):
        for s in ctx["sources"]:
            if isinstance(s, str):
                s_norm = s.strip().lower()
                sources.add(s_norm)
                m = re.match(r"^(?:doc)?(\d+)$", s_norm)
                if m:
                    sources.add(m.group(1))
    unsupported = []
    for c in cites:
        if c.startswith("http"):
            if not any(s in c or c in s for s in sources):
                unsupported.append(c)
        else:
            if sources and (c not in sources):
                unsupported.append(c)
    return {
        "missing_citations": len(cites) == 0,
        "unsupported_citations": unsupported,
        "citation_count": len(cites),
    }

def multi_ref_reduce(metric_fn, prediction: str, references: List[str]) -> float:
    """Apply a scalar metric against multiple references (take max over refs)."""
    if not references:
        return 0.0
    return max(metric_fn(prediction, r) for r in references)
EOF
