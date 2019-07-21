# indentation must be taps
TEST_PATH="./tests"
MUSHROOM_DEST="./datasets/mushroom/mushroom.csv"
MUSHROOM_SOURCE="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

init:
	pip install -r requirements.txt

test: clean-pyc
	py.test --verbose --color=yes $(TEST_PATH)

lint:
	flake8 --exclude=venv/

fetch-data:
	wget -O $(MUSHROOM_DEST) $(MUSHROOM_SOURCE)
clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete

.PHONY: clean-pyc
