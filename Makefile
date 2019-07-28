# indentation must be taps
APP_NAME="cb-recommender"
TEST_PATH="./tests"
MUSHROOM_DEST="./datasets/mushroom/mushroom.csv"
MUSHROOM_SOURCE="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
WINDOW=512

N_TRIALS=3
N_ROUNDS=10000

init:
	pip install -r requirements.txt

test: clean-pyc
	py.test --verbose --color=yes $(TEST_PATH)

lint:
	flake8 --exclude=venv/

run:
	python main.py "synthetic" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS)
	python main.py "mushroom" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS)
	python main.py "news" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS) --is_acp --grad_clip

plot:
	python plotting.py "synthetic" --n_trials $(N_TRIALS) --window $(WINDOW)
	python plotting.py "mushroom" --n_trials $(N_TRIALS) --window $(WINDOW)
	python plotting.py "news" --n_trials $(N_TRIALS) --window $(WINDOW)

fetch-data:
	wget -O $(MUSHROOM_DEST) $(MUSHROOM_SOURCE)

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete


docker-run:
	docker build \
      --file=./Dockerfile \
      --tag=$(APP_NAME) ./
	docker run \
      --detach=false \
      --name=$(APP_NAME) \
      --publish=$(HOST):8080 \
      my_project

.PHONY: clean-pyc
