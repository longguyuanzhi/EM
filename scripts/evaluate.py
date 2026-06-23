from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out", default="runs/summary.csv")
    args = p.parse_args()
    rows = []
    for path in args.results:
        obj = json.loads(Path(path).read_text())
        row = {"path": path, "benchmark": obj.get("benchmark"), "seed": obj.get("seed"), "n_train_final": obj.get("n_train_final")}
        row.update(obj["final_metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df)


if __name__ == "__main__":
    main()
