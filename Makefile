init:
	pip install -r requirements.txt
test:
	bash main.sh
	py.test tests
