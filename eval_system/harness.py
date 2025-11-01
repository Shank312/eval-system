import argparse
import json
import os
from typing import Dict, List, Any

from eval_system.metrics import (
    exact_match, f1_score, rouge_scores,
    hallucination_check, multi_ref_reduce
)

def _load_mapping(path: str, key_field: str, value_field: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    out = {}
    for row in data:
        if key_field not in row:
            raise ValueError(f"Missing '{key_field}' in row: {row}")
        out[row[key_field]] = row[value_field]
    return out

def evaluate(preds: Dict[str, str], refs: Dict[str, List[str]], ctx: Dict[str, Any] | None) -> Dict[str, Any]:
    ids = sorted(set(preds.keys()) & set(refs.keys()))
    per_example = {}
    agg = {
        "exact_match": 0.0,
        "f1": 0.0,
        "rouge1": 0.0,
        "rougeL": 0.0,
        "missing_citations_rate": 0.0,
        "unsupported_citations_rate": 0.0,
        "avg_citation_count": 0.0,
    }
    for i in ids:
        p = preds[i]
        rlist = refs[i]
        if isinstance(rlist, str):
            rlist = [rlist]
        em = multi_ref_reduce(lambda a,b: exact_match(a,b), p, rlist)
        f1 = multi_ref_reduce(lambda a,b: f1_score(a,b), p, rlist)
        r_scores = rouge_scores(p, rlist)
        cinfo = hallucination_check(p, (ctx or {}).get(i) if ctx else None)
        per_example[i] = {
            "prediction": p,
            "references": rlist,
            "metrics": {
                "exact_match": em,
                "f1": f1,
                "rouge1": r_scores["rouge1"],
                "rougeL": r_scores["rougeL"],
                "missing_citations": cinfo["missing_citations"],
                "unsupported_citations": cinfo["unsupported_citations"],
                "citation_count": cinfo["citation_count"],
            }
        }
        agg["exact_match"] += em
        agg["f1"] += f1
        agg["rouge1"] += r_scores["rouge1"]
        agg["rougeL"] += r_scores["rougeL"]
        agg["missing_citations_rate"] += 1.0 if cinfo["missing_citations"] else 0.0
        agg["unsupported_citations_rate"] += 1.0 if cinfo["unsupported_citations"] else 0.0
        agg["avg_citation_count"] += cinfo["citation_count"]
    n = max(len(ids), 1)
    for k in ["exact_match","f1","rouge1","rougeL","missing_citations_rate","unsupported_citations_rate","avg_citation_count"]:
        agg[k] = agg[k] / n
    return {"overall": agg, "count": len(ids), "per_example": per_example}

def main():
    ap = argparse.ArgumentParser(description="LLM Eval Harness")
    ap.add_argument("--pred", required=True, help="Path to preds.json (dict id->str or list of {id,prediction})")
    ap.add_argument("--refs", required=True, help="Path to refs.json (dict id->list[str] or list of {id,references})")
    ap.add_argument("--ctx", default=None, help="Optional path to ctx.json (dict id->{sources:[...]})")
    ap.add_argument("--out", default=None, help="Optional output JSON path; prints to stdout if omitted")
    args = ap.parse_args()
    preds = _load_mapping(args.pred, "id", "prediction")
    refs_raw = _load_mapping(args.refs, "id", "references")
    refs = {k: (v if isinstance(v, list) else [v]) for k, v in refs_raw.items()}
    ctx = None
    if args.ctx and os.path.exists(args.ctx):
        with open(args.ctx, "r", encoding="utf-8") as f:
            ctx = json.load(f)
    results = evaluate(preds, refs, ctx)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
