"""BBQ (Bias Benchmark for Question Answering) evaluation package."""

from evaluation.bbq.data import get_bbq_data
from evaluation.bbq.metrics import compute_bbq_metrics

__all__ = ["get_bbq_data", "compute_bbq_metrics"]
