[![Build Status](https://travis-ci.com/dhfromkorea/contextual-bandit-recommender.svg?token=LpCqnxSYFM2Cg2x3ixjz&branch=master)](https://travis-ci.com/dhfromkorea/contextual-bandit-recommender)

# goal
price quote recommendation
discretize price space?

##

The problem:
T-stream of contexts, learn to recommend a_t.
assumes no labels and a reward function (in some applications, there is a natural ... such as CTR).

Examples of such scenarios include online advertising, where we only know whether a user clicked an ad that he was presented with, but don't know which other ads he would have clicked; or clinic trials where we know how a person responded to a treatment, but don't know how he would have responded to a different treatment.

While, in general, algorithms for the contextual bandits problem assume continuous rewards in the range [0,1], this package deals only with the case of discrete rewards {0,1}, and only with the case of arms that all see the same covariates.

##
- fix only measured inefficiencies (as a protection against premature optimization)

## reproduce
https://arxiv.org/pdf/1003.0146.pdf
https://arxiv.org/pdf/1802.09127.pdf
http://www.cs.cmu.edu/~lizhou/papers/LCB_IJCAI16.pdf
https://arxiv.org/pdf/1811.04383.pdf
(read cold start)

(check out)
https://github.com/david-cortes/contextualbandits

## tech
eval metrics
(diag)
- simple reward: 500 steps average
- cum rewards: over T steps
- action value estimates: snapshot every n steps (x axis: actions, y: value)

(eval)
- simple regret
- cum regrets
- regret: over time (peak)

(maybe)
- simple regret per action
- cum regrets per action

log the file as a single np.array?

want to understand
- performance
- diagnostics on a possible cause: bias or variance?

### highest priority
[ ] some visualization/diagnostics (eval metrics)
[ ] pull next data: news data
[ ] write new_data sampler
[ ] write test for it
[ ] run diagnostics and fix bugs in models (bias term issue)
[ ] AWS (so fast --- all tests passing)

### mid priority
[ ] Blogpost
[ ] ec2 & s3
[ ] pytorch or tensorflow

### low priority
[ ] logging
[ ] SQL - Data model?
[ ] deal with large data
[ ] spark or hadoop
[ ] Docker Container
[v] test
[v] build/makefile

## TODO
[v] do eda on mushroom data
[v] write mushroom data sampler
[v] write basic policies for mushroom
[v] write tests for methods so far
[v] decouple tests from real datasets
[v] write a Linear-Gaussian bandit sampler?
[v] write a basic thompson sampling method
[v] write a test for thompson sampling (beta bernoulli)
[v] add synthetic data
[v] consider using mock (at this point no!, arguing from premature optimization)
[ ] pull next data: jester
[ ] pull next data: movielens
[ ] pull next data: goodbooks
[ ] pull next data: price recommendation
[ ] write a hybrid method
[ ] write a diagnostic, loss, evaluation workflow (plots)


## questions
- why not supervised learning if history data is available?
- at least pretrainining in the batch setting, possible?
- disjoint model (for each action) is bad, bad computational cost (shared components ignored)
- build a joint model that computes with input x_t, a_t no?
- cold start problem

## start simple
- write tests
- write code
- check performance

- (next) big dataset

### Contextual bandit
reproduce
- [ ] thompson bandit
- [ ] deep bayesian
- [ ] latent contextual model

https://arxiv.org/pdf/1003.0146.pdf
https://arxiv.org/pdf/1802.09127.pdf
http://www.cs.cmu.edu/~lizhou/papers/LCB_IJCAI16.pdf
https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf

### professional
https://github.com/bfortuner/vaa3d-api



http://courses.cms.caltech.edu/cs101.2/slides/cs101.2-02-Bandits-notes.pdf
https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
http://chercheurs.lille.inria.fr/~ghavamza/RL-EC-Lille/Lecture%20Bandit.pdf


### RL
https://gdmarmerola.github.io/ts-for-mushroom-bandit/
http://jmlr.csail.mit.edu/papers/volume6/shani05a/shani05a.pdf
https://arxiv.org/pdf/1810.12027.pdf

### Survey
https://arxiv.org/pdf/1707.07435.pdf

### references
https://github.com/maciejkula/spotlight
https://github.com/ntucllab/striatum
https://github.com/KKeishiro/Yahoo_recommendation/tree/master/data
https://github.com/thunfischtoast/LinUCB
https://github.com/nicku33/demo/blob/master/Contextual%20Bandit%20Synthetic%20Data%20using%20LinUCB.ipynb
https://github.com/allenday/contextual-bandit
https://github.com/david-cortes/contextualbandits

### Data
https://www.kdnuggets.com/2016/02/nine-datasets-investigating-recommender-systems.html
http://cseweb.ucsd.edu/~jmcauley/datasets.html
https://github.com/caserec/Datasets-for-Recommneder-Systems

### Kaggle kernels
https://www.kaggle.com/morrisb/how-to-recommend-anything-deep-recommender
https://www.kaggle.com/skillsmuggler/what-do-you-recommend
https://www.kaggle.com/rblcoder/recommend-based-on-nearest-neighbors
https://www.kaggle.com/CooperUnion/anime-recommendations-database
https://www.kaggle.com/tamber/steam-video-games
https://www.kaggle.com/alexattia/the-simpsons-characters-dataset
https://www.kaggle.com/jrobischon/wikipedia-movie-plots
https://www.kaggle.com/rmisra/clothing-fit-dataset-for-size-recommendation
https://blog.insightdatascience.com/tunable-and-explainable-recommender-systems-cd52b6287bad

### course
http://web.stanford.edu/class/cs246/index.html#schedule
https://hci.stanford.edu/courses/cs448g/a2/files/map_reduce_tutorial.pdf
http://web.stanford.edu/class/cs246/slides/07-recsys1.pdf
http://web.stanford.edu/class/cs246/slides/08-recsys2.pdf


### data science tips
https://elitedatascience.com/python-data-wrangling-tutorial
