# indentation must be taps
TEST_PATH="./tests"

init:
	pip install -r requirements.txt

test: clean-pyc
	py.test --verbose --color=yes $(TEST_PATH)

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete

.PHONY: clean-pyc
