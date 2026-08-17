"""
src/evaluation/__init__.py
==========================
Lazy-import public API so that importing sub-modules like
`src.evaluation.metrics` (pure Python, no external deps) does NOT
eagerly load the runner — which would pull in the full retriever/cohere stack.
"""

__all__ = ["run_evaluation", "load_qrels", "print_report"]


def __getattr__(name):
    if name in ("run_evaluation", "load_qrels", "print_report"):
        from src.evaluation import runner as _runner
        return getattr(_runner, name)
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")
