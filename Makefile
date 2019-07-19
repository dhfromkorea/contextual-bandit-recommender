# indentation must be taps
TEST_PATH="./tests"

init:
	pip install -r requirements.txt

test: clean-pyc
	py.test --verbose --color=yes $(TEST_PATH)

lint:
	flake8 --exclude=venv/

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete

.PHONY: clean-pyc
