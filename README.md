# Eval System (MVP)

Automated evaluation harness for LLM outputs 🚀

This evaluates predicted responses from LLMs against reference answers and checks for hallucination / citation violations.

Supports:
- Exact Match
- Token-Level F1
- ROUGE-1 / ROUGE-L
- Simple hallucination/citation check

---

## Install

```bash
git clone https://github.com/Shank312/eval-system
cd eval-system

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Git Bash:
source .venv/Scripts/activate

pip install -r requirements.txt

python - << "EOF"
import nltk
nltk.download("punkt", quiet=True)
EOF


Run Evaluation

Put your predictions, references, and optional context into JSON.

Example files are in /examples

Run evaluation:
PYTHONPATH=. python -m eval_system.harness \
  --pred examples/preds.json \
  --refs examples/refs.json \
  --ctx  examples/ctx.json \
  --out  examples/results.json
Output → examples/results.json (per-example + overall metrics)


JSON Formats

preds.json
[
  { "id": "ex1", "prediction": "Paris is the capital of France. [1]" }
]

refs.json
[
  { "id": "ex1", "references": ["Paris is the capital of France."] }
]

ctx.json
{
  "ex1": { "sources": ["1"] }
}


Why this exists

In production — LLMs need evaluation systems.

This project is a minimal infra version:

convert predictions → structured JSON metrics

detect missing citations

catch unsupported references

report precision/recall style scores

It is easy to integrate with CI / batch eval.


Why this exists

In production — LLMs need evaluation systems.

This project is a minimal infra version:

convert predictions → structured JSON metrics

detect missing citations

catch unsupported references

report precision/recall style scores

It is easy to integrate with CI / batch eval.


Roadmap

BLEU scoring

Constraint checker (regex rules)

PDF report export

Web dashboard


Author

Built by Shankar Kumar
Learning-in-public | Evaluations | LLM Infra

GitHub: https://github.com/Shank312
