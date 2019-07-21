# indentation must be taps
TEST_PATH="./tests"
MUSHROOM_PATH="./datasets/mushroom"

init:
	pip install -r requirements.txt

test: clean-pyc
	py.test --verbose --color=yes $(TEST_PATH)

lint:
	flake8 --exclude=venv/

fetch-data:
	wget -O "$(MUSHROOM_PATH)/mushroom.csv" "https://www.kaggle.com/uciml/mushroom-classification/downloads/mushrooms.csv/1"

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete

.PHONY: clean-pyc
