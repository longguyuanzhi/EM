from __future__ import annotations

import argparse

from activeem.benchmarks import get_benchmark_spec
from activeem.data import make_synthetic_dataset, save_npz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["filter", "resonator", "coupler"], required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    spec = get_benchmark_spec(args.benchmark)
    data = make_synthetic_dataset(spec, args.n, args.seed, args.noise)
    save_npz(args.out, data)
    print(f"Saved {args.benchmark} synthetic dataset to {args.out}")


if __name__ == "__main__":
    main()
