# Additional native WarBias metrics

[Results](results.json) contain the six fixed scale-one task-vector panels.
[Input verification](input_verification.json) binds the baseline development
predictions, source data and calculation code. [Development profiles](development_profiles.json)
record the matched numerical execution profiles. The archive source mapping is
in [publication.json](../reports/publication.json).

NNR uses the existing development q20 rule, with q10/q30 sensitivity checks.
Each model has only two development examples from one independent component.
These estimates are exploratory, and their intervals are conditional on tau.
The eligible cohort is fixed from each profile's baseline. Post-intervention
outcomes do not select tau or the intervention strength.

M1, M1b and M2 use native LMS on matched gender and age panels. The other
demographic attribute is held fixed. Each group requires five independent
components. M1 is minimum post-intervention group quality. M1b is the minimum
divided by the maximum. M2 reports the range and population standard deviation
of paired group quality losses. Intervals use 1,000 paired component bootstrap
draws with seed 42.

Language RI uses UK as target and EN as non-target. Harm is the absolute
distance of component-macro SS from 50. The ratio requires target harm reduction
above one percentage point and a positive paired 95% lower bound. No current
model passes that rule. All six ratios remain null; their numerators,
denominators and intervals are retained.

These WarBias diagnostics do not replace the full-study protocol inventory.
RMU and WGA values remain in their original records.
