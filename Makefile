.PHONY: clean clean-test clean-pyc clean-build docs help
init:
	pip install -r requirements.txt

test:
	pytest tests


docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/sugiyama.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ sugiyama
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html