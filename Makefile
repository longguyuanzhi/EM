.PHONY: install test quickstart smoke tables clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

quickstart:
	python examples/quickstart.py

smoke:
	python scripts/run_active_learning.py --config configs/filter.yaml --rounds 1 --initial 16 --batch 4 --candidate-pool 120 --synthetic --out runs/filter_smoke

tables:
	python scripts/reproduce_tables.py --configs configs/filter.yaml configs/resonator.yaml configs/coupler.yaml --synthetic --seeds 0 1 --out runs/rebuttal_tables_synthetic

clean:
	rm -rf runs __pycache__ .pytest_cache activeem.egg-info
