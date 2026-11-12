"""
Tests for RAGAS evaluation harness and RAG triad metrics (Day 10).

Covers:
- Evaluation sample scoring (faithfulness, answer relevancy, precision, recall)
- Unanswerable query disciplined abstention
- Cosine similarity calculation edge cases
- Golden set format integrity
"""

import pytest
from pathlib import Path

from eval.benchmark_ragas import (
    evaluate_sample,
    compute_cosine_similarity,
    load_golden_set,
)


class TestRagasEvaluationHarness:
    """Unit tests for the RAGAS financial evaluation harness."""

    def test_compute_cosine_similarity(self):
        """Verify vector cosine similarity calculation across edge cases."""
        # Identical vectors -> 1.0
        vec = [0.5, 0.5, 0.5, 0.5]
        assert pytest.approx(compute_cosine_similarity(vec, vec), 0.001) == 1.0

        # Orthogonal vectors -> 0.0
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert pytest.approx(compute_cosine_similarity(vec_a, vec_b), 0.001) == 0.0

    def test_evaluate_sample_answerable(self):
        """Verify triad metrics for answerable financial query."""
        sample_item = {
            "question": "What was Apple's total revenue in fiscal year 2023?",
            "expected_answer": "Apple reported total revenue of approximately $383.3 billion in fiscal year 2023.",
            "expected_chunks": [],
            "ticker": "AAPL",
            "doc_type": "10-K",
            "answerable": True,
            "difficulty": "easy",
            "category": "financial_metric",
        }
        res = evaluate_sample(sample_item)

        assert res["is_answerable"] is True
        assert res["decision"] == "generate"
        assert res["faithfulness"] == 1.0
        assert 0.0 <= res["answer_relevancy"] <= 1.0
        assert res["context_precision"] == 1.0
        assert res["context_recall"] == 1.0

    def test_evaluate_sample_unanswerable(self):
        """Verify safe abstention and zero hallucination on unanswerable query."""
        unanswerable_item = {
            "question": "What will Apple's stock price be in fiscal 2050?",
            "expected_answer": "",
            "expected_chunks": [],
            "ticker": "AAPL",
            "doc_type": "10-K",
            "answerable": False,
            "difficulty": "hard",
            "category": "speculative",
        }
        res = evaluate_sample(unanswerable_item)

        assert res["is_answerable"] is False
        assert res["decision"] == "abstain"
        assert res["abstain_correct"] is True
        assert res["faithfulness"] == 1.0  # Safe abstention is faithful

    def test_golden_set_loading(self):
        """Verify the 119-question golden set loads and contains required schema keys."""
        golden_path = Path(__file__).parent.parent / "eval" / "golden_set.jsonl"
        items = load_golden_set(golden_path)

        assert len(items) == 119
        for item in items:
            assert "question" in item
            assert "answerable" in item
            assert "category" in item
            assert "ticker" in item
