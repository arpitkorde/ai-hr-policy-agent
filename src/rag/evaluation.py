"""RAG Evaluation using RAGAS framework.

Measures retrieval and generation quality to ensure the HR agent
provides accurate, grounded answers. Supports automated evaluation
for continuous monitoring.

Metrics:
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are the retrieved chunks actually relevant?
"""

import logging
from dataclasses import dataclass

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured evaluation result."""

    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    overall_score: float
    num_samples: int


class RAGEvaluator:
    """Evaluates RAG pipeline quality using RAGAS metrics.

    RAGAS (Retrieval Augmented Generation Assessment) provides
    reference-free evaluation — no ground truth answers needed.
    """

    def __init__(self):
        self.metrics = [faithfulness, answer_relevancy, context_precision]
        logger.info("RAGAS evaluator initialized with metrics: "
                     "faithfulness, answer_relevancy, context_precision")

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> EvaluationResult:
        """Run RAGAS evaluation on a set of Q&A pairs.

        Args:
            questions: List of user questions.
            answers: List of generated answers.
            contexts: List of retrieved context chunks (list of lists).
            ground_truths: Optional list of expected answers.

        Returns:
            EvaluationResult with scores for each metric.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        logger.info(f"Running RAGAS evaluation on {len(questions)} samples...")
        results = evaluate(dataset, metrics=self.metrics)

        eval_result = EvaluationResult(
            faithfulness_score=float(results.get("faithfulness", 0)),
            answer_relevancy_score=float(results.get("answer_relevancy", 0)),
            context_precision_score=float(results.get("context_precision", 0)),
            overall_score=float(
                (results.get("faithfulness", 0)
                 + results.get("answer_relevancy", 0)
                 + results.get("context_precision", 0)) / 3
            ),
            num_samples=len(questions),
        )

        logger.info(
            f"Evaluation complete | faithfulness={eval_result.faithfulness_score:.3f} | "
            f"relevancy={eval_result.answer_relevancy_score:.3f} | "
            f"precision={eval_result.context_precision_score:.3f} | "
            f"overall={eval_result.overall_score:.3f}"
        )

        return eval_result

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult:
        """Evaluate a single Q&A pair.

        Convenience method for testing individual queries.
        """
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
        )
