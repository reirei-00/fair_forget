"""Component-weighted estimation, the component bootstrap and rank correlation.

Claim components are the independent sampling units: rows inside a component
(profiles, translations) are averaged first, then components get equal weight.
"""

from collections import defaultdict
from statistics import fmean

import numpy as np

from lm_eval.benchmark.warbias.schema import value


def component_mean(items, field):
    """Mean of ``field`` within each component, then across components.

    Items whose value is None (abstentions, missing scores) are skipped; None
    when no component has a value.
    """
    by_cluster = defaultdict(list)
    for item in items:
        observed = value(item, field)
        if observed is not None:
            by_cluster[value(item, "cluster")].append(observed)
    return fmean(fmean(v) for v in by_cluster.values()) if by_cluster else None


def components(items):
    return len({value(item, "cluster") for item in items})


def bootstrap_draws(clusters, draws, seed):
    """Component resamples (with replacement); empty when no interval is defined."""
    clusters = sorted(clusters)
    if draws <= 0 or len(clusters) < 2:
        return []
    rng = np.random.default_rng(seed)
    return [list(rng.choice(clusters, size=len(clusters))) for _ in range(draws)]


class _Relabelled:
    """A resampled item: its own component label, every other field delegated."""

    __slots__ = ("_item", "cluster")

    def __init__(self, item, cluster):
        self._item = item
        self.cluster = cluster

    def __getattr__(self, name):
        return getattr(self._item, name)


def _relabel(item, label):
    if isinstance(item, dict):
        return {**item, "cluster": label}
    return _Relabelled(item, label)


def resample(items, draw):
    """Items of one bootstrap draw; a component drawn twice counts twice."""
    by_cluster = defaultdict(list)
    for item in items:
        by_cluster[value(item, "cluster")].append(item)
    return [
        _relabel(item, f"{k}:{cluster}")
        for k, cluster in enumerate(draw)
        for item in by_cluster[cluster]
    ]


def percentile_interval(values):
    values = [v for v in values if v is not None]
    return np.quantile(values, [0.025, 0.975]).tolist() if values else None


def component_bootstrap(items, statistic, draws, seed):
    """95% percentile interval of ``statistic`` over whole-component resamples."""
    samples = bootstrap_draws({value(i, "cluster") for i in items}, draws, seed)
    if not samples:
        return None
    return percentile_interval([statistic(resample(items, d)) for d in samples])


def _ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ordered = np.asarray(values, dtype=float)[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2
        start = end + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation with average ranks; None when undefined."""
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
