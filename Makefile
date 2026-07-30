.PHONY: verify preflight figures

verify:
	python3 reproduction/tests/verify_release.py
	python3 reproduction/tests/check_public_tree.py

preflight:
	python3 reproduction/scripts/preflight.py --mode full

figures:
	python3 reproduction/scripts/make_public_figures.py
