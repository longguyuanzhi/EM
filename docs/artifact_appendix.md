# Artifact Appendix

This document maps artifact components to rebuttal/revision claims.

| Reviewer | Concern | Artifact location |
|---|---|---|
| 1065A | uncertainty calibration | `activeem/uncertainty.py`, `activeem/metrics.py` |
| 1065A | response-domain physics loss | `activeem/losses.py` |
| 1065A | concrete RF design outcome | `scripts/optimize_design.py`, `activeem/optimize.py` |
| 1065B | reproducible benchmarks | `configs/*.yaml`, `activeem/benchmarks.py` |
| 1065B | FNO input construction | `activeem/encoding.py`, `activeem/models.py` |
| 1065B | more seeds/statistics | `scripts/reproduce_tables.py` |
| 1065C | acquisition details | `activeem/acquisition.py` |
| 1065C | safe filtering | `activeem/acquisition.py::calibrated_safe_mask` |
| 1065D | anonymous code | full package |

The synthetic solver is for code validation only. Replace `SyntheticEMSolver`
with `ExternalFEMSolver` for final FEM-backed experiments.
