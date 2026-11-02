"""WarBias detection pipeline: annotated triplet panel, scoring and analysis.

Layers, in data-flow order:

- ``prepare``: build the pinned, adapted panel bundle from the released triplets.
- ``panel``: load and verify saved runs for the whole panel (coverage, identity).
- ``scoring``: turn saved likelihoods into one ``TripletItem`` per triplet
  (raw length scorings and the prior-normalised stereotype preference).
- ``stats`` / ``aggregate``: component-weighted means, the component bootstrap
  and every group analysis; they work on any ``TripletItem`` source.
- ``detection``: the detection report assembled from the layers above.
- ``calibration``: the no-preference band tau and the group-support rule.
- ``tables``: paper tables rendered from saved reports, without recomputation.
- ``intervention``: matched original/neutral checkpoint effects and transfer.

Source creation and case annotation are maintained in the the-project repository.

Rates are stored in [0, 1] in every JSON output; tables render percentages.
Margins are in nats.
"""
