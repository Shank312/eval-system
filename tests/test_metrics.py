

cat > tests/test_metrics.py << 'EOF'
import math
from eval_system.metrics import exact_match, f1_score, rouge_scores, hallucination_check, extract_citations

def test_exact_match_basic():
    assert exact_match("Hello, World!", "hello world") == 1
    assert exact_match("Hello", "Hi") == 0

def test_f1_score_tokens():
    assert math.isclose(f1_score("a b c", "a b c"), 1.0)
    assert math.isclose(f1_score("a b", "a b c"), 0.8)
    assert math.isclose(f1_score("", ""), 1.0)
    assert math.isclose(f1_score("", "nonempty"), 0.0)

def test_rouge_scores_runs():
    s = rouge_scores("the quick brown fox", ["the quick fox", "a brown dog"])
    assert "rouge1" in s and "rougeL" in s
    assert 0.0 <= s["rouge1"] <= 1.0
    assert 0.0 <= s["rougeL"] <= 1.0

def test_citation_detection_and_check():
    text = "Answer per [1] and [doc2]. Also see https://example.com (Source: ABC)."
    cites = extract_citations(text)
    assert any(c in cites for c in ["1", "doc2"])
    info = hallucination_check(text, {"sources": ["1", "doc2", "https://example.com/page"]})
    assert info["citation_count"] >= 3
    assert info["missing_citations"] is False
    assert info["unsupported_citations"] == []

def test_citation_missing():
    info = hallucination_check("This is an answer with no citations.")
    assert info["missing_citations"] is True
EOF
