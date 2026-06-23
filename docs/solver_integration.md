# External Solver Integration

`ExternalFEMSolver` connects ActiveEM to a commercial or internal simulation flow.

Example:

```bash
python scripts/run_active_learning.py   --config configs/filter.yaml   --solver-command "python my_solver_wrapper.py {request} {response}"   --out runs/filter_fem
```

The wrapper receives:

```text
request.json response.npz
```

## Request JSON schema

```json
{
  "benchmark": "filter",
  "x_norm": [[...]],
  "freq_grid_ghz": [...],
  "variables": [
    {"name": "L1", "low": 15.0, "high": 19.0, "unit": "mm"}
  ]
}
```

## Response NPZ schema

```text
spectra: float32 array [N, P, F]
scalars: float32 array [N, S]
```

Recommended wrapper steps:

1. Convert normalized parameters to physical units.
2. Generate layout/geometry.
3. Run full-wave simulation.
4. Extract response curves and scalar metrics.
5. Save `response.npz`.
6. Store raw solver logs separately.
