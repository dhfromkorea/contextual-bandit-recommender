# indentation must be taps
TEST_PATH="./tests"
MUSHROOM_DEST="./datasets/mushroom/mushroom.csv"
MUSHROOM_SOURCE="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

N_TRIALS=3
N_ROUNDS=50000

init:
	pip install -r requirements.txt

test: clean-pyc
	py.test --verbose --color=yes $(TEST_PATH)

lint:
	flake8 --exclude=venv/

run:
	python main.py "news" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS) --is_acp
	python main.py "synthetic" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS)
	python main.py "mushroom" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS)

plot:
	python plotting.py --n_trials $(N_TRIALS)

fetch-data:
	wget -O $(MUSHROOM_DEST) $(MUSHROOM_SOURCE)

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete

.PHONY: clean-pyc
