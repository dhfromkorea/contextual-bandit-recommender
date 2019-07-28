[![Build Status](https://travis-ci.com/dhfromkorea/contextual-bandit-recommender.svg?token=LpCqnxSYFM2Cg2x3ixjz&branch=master)](https://travis-ci.com/dhfromkorea/contextual-bandit-recommender)

# Recommendater System with Contextual Bandit Algorithms

This repo contains a work-in-progress code for the implementations of commmon contextual bandit algorithms. 


## Getting Started


### Prerequisites
Built for python 3.5+. 
```
numpy==1.16.4
pandas==0.25.0
scikit-learn==0.21.2
scipy==1.3.0
seaborn==0.9.0
sklearn==0.0
torch==1.1.0
```

To install prerequisites, preferably in a virtualenv or similiar.
```
make init
```
### Running

Change the experiment parameters in `Makefile`.

```
make run
```

or tune the hyperparameters yourself (check the args in `main.py`). 

```
python main.py "synthetic" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS)
python main.py "mushroom" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS)
python main.py "news" --n_trials $(N_TRIALS) --n_rounds $(N_ROUNDS) --is_acp --grad_clip
```

### Available Algorithms
* LinUCB: Linear UCB algorithm (modified [1]).
* Thompson Sampling: Linear Gaussian with a conjugate prior [2].
* Neural Network Policy: A fully-connected neural network with gradient noise.
* Epsilon Greedy
* UCB policy
* Sample Mean Policy
* Random Policy


### Datasets
Check out the [blogpost](http://www.dhfromkorea.com/news-recommendation-with-contextual-bandit/) for the details about the datasets

- Synthetic: available built-in.
- Mushroom: run `make fetch-data`.
- Yahoo Front Page Click Log Data: need to make a request.


### Running the tests
```
make test
```

### References
[1]: http://rob.schapire.net/papers/www10.pdf
[2]: https://en.wikipedia.org/wiki/Bayesian_linear_regression

