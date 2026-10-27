"""Explicit development-only swapping for the BiasUnlearn-style adaptation."""

import math


class SwapController:
    def __init__(self, *, metric, threshold, comparison, swap_pairs):
        if not metric or threshold is None or not math.isfinite(threshold):
            raise ValueError("A controller requires a named metric and explicit finite threshold")
        if comparison not in {"ge", "le"} or not swap_pairs:
            raise ValueError("A controller needs a comparison and declared target pairs")
        self.metric, self.threshold, self.comparison = metric, threshold, comparison
        self.swap_pairs = tuple(tuple(pair) for pair in swap_pairs)
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in self.swap_pairs):
            raise ValueError("Controller swaps require distinct forget/correction example pairs")
        flattened = [item for pair in self.swap_pairs for item in pair]
        if any(not isinstance(item, str) or not item for item in flattened) or len(
            set(flattened)
        ) != len(flattened):
            raise ValueError("Controller swap pairs must have unique nonempty example IDs")
        self.active = False

    def update(self, evidence):
        if evidence.get("split") != "method_development":
            raise ValueError("Swapping decisions require method-development evidence")
        value = evidence.get(self.metric)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Controller metric is missing or nonfinite")
        self.active = (
            value >= self.threshold if self.comparison == "ge" else value <= self.threshold
        )
        return self.active

    def replacement(self, example_id):
        if self.active:
            for first, second in self.swap_pairs:
                if example_id == first:
                    return second
                if example_id == second:
                    return first
        return example_id
