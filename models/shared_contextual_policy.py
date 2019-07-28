"""
The policies that share parameters across all actions.

The current implementations accept both user and item context.
"""

import sys

import numpy as np
from scipy.stats import invgamma

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import logging
logger = logging.getLogger(__name__ + "shared_contextual_policy")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class SharedLinUCBPolicy(object):
    """LinUCB policy that shares parameters across all actions.

    Note this is different from the hybrid model in [1].

    [1]: https://arxiv.org/pdf/1003.0146.pdf
    """
    def __init__(self, context_dim, delta=0.2,
                 train_starts_at=500, train_freq=50,
                 batch_mode=False, batch_size=1024
                 ):

        # bias
        self._d = context_dim + 1

        # initialize with I_d, 0_d
        self._A = np.identity(self._d)
        # we will update this every train_freq
        self._A_inv = np.linalg.inv(self._A)

        self._b = np.zeros(self._d)
        self._theta = np.linalg.inv(self._A).dot(self._b)

        self._alpha = 1 + np.sqrt(np.log(2/delta)/2)

        self._t = 0
        self._train_freq = train_freq
        self._train_starts_at = train_starts_at

        # for computational efficiency
        # train on a random subset
        self._batch_mode = batch_mode
        self._batch_size = batch_size

    def choose_action(self, x_t):
        u_t, S_t = x_t
        # number of actions can change
        n_actions = S_t.shape[0]

        # estimate an action value
        Q = np.zeros(n_actions)
        ubc_t = np.zeros(n_actions)

        for j in range(n_actions):
            # compute input for each action
            # user_context + action_context + bias
            x_t = np.concatenate( (u_t, S_t[j, :], [1]) )
            assert len(x_t) == self._d

            # compute upper bound
            k_ta = x_t.T.dot(self._A_inv).dot(x_t)
            ubc_t[j] = self._alpha * np.sqrt(k_ta)
            Q[j] = self._theta.dot(x_t) + ubc_t[j]

        a_t = np.argmax(Q)

        self._t += 1

        return a_t

    def update(self, a_t, x_t, r_t):
        u_t, S_t = x_t

        x_t = np.concatenate( (u_t, S_t[a_t, :], [1]) )
        assert len(x_t) == self._d

        # d x d
        self._A += x_t.dot(x_t.T)
        # d x 1
        self._b += r_t * x_t

        if self._t < self._train_starts_at:
            return

        if self._t % self._train_freq == 0:
            self._A_inv = np.linalg.inv(self._A)
            self._theta = self._A_inv.dot(self._b)


class SharedLinearGaussianThompsonSamplingPolicy(object):
    """Linear Gaussian Thompson Sampling policy that shares parameters
    across all actions.


    A Bayesian approach for inferring the true reward distribution model.

    Implements a Bayesian linear regression with a conjugate prior [1].

    Model: Gaussian: R_t = W*x_t + eps, eps ~ N(0, sigma^2 I)
    Model Parameters: mu, cov
    Prior on the parameters
    p(w, sigma^2) = p(w|sigma^2) * p(sigma^2)

    1. p(sigma^2) ~ Inverse Gamma(a_0, b_0)
    2. p(w|sigma^2) ~ N(mu_0, sigma^2 * precision_0^-1)

    For computational reasons,

    1. model updates are done periodically.
    2. for large sample cases, batch_mode is enbabled.

    [1]: https://en.wikipedia.org/wiki/Bayesian_linear_regression#Conjugate_prior_distribution

    """

    def __init__(self,
                 context_dim,
                 eta_prior=6.0,
                 lambda_prior=0.25,
                 train_starts_at=500,
                 posterior_update_freq=50,
                 batch_mode=True,
                 batch_size=256,
                 lr=0.01):
        self._t = 1
        self._update_freq = posterior_update_freq
        self._train_starts_at = train_starts_at

        # bias
        self._d = context_dim + 1

        # inverse gamma prior
        self._a_0 = eta_prior
        self._b_0 = eta_prior
        self._a = eta_prior
        self._b = eta_prior

        # conditional Gaussian prior
        self._sigma_sq_0 = invgamma.rvs(eta_prior, eta_prior)
        self._lambda_prior = lambda_prior
        # precision_0 shared for all actions
        self._precision_0 = self._sigma_sq_0 / self._lambda_prior

        # initialized at mu_0
        self._mu = np.zeros(self._d)

        # initialized at cov_0
        self._cov = 1.0 / self._lambda_prior * np.eye(self._d)

        # remember training data
        self._X_t = None

        # for computational efficiency
        # train on a random subset
        self._batch_mode = batch_mode
        self._batch_size = batch_size
        self._lr = lr


    def _update_posterior(self, X_t, r_t_list):
        cov_t = np.linalg.inv(np.dot(X_t.T, X_t) + self._precision_0)
        mu_t = np.dot(cov_t, np.dot(X_t.T, r_t_list))
        a_t = self._a_0 + self._t/2

        # mu_0 simplifies some terms
        r = np.dot(r_t_list, r_t_list)
        precision_t = np.linalg.inv(cov_t)
        b_t = self._b_0 + 0.5*(r - np.dot(mu_t.T, np.dot(precision_t, mu_t)))

        if self._batch_mode:
            # learn bit by bit
            self._cov = cov_t * self._lr + self._cov * (1 - self._lr)
            self._mu = mu_t * self._lr + self._cov * (1 - self._lr)
            self._a = a_t * self._lr + self._cov * (1 - self._lr)
            self._b = b_t * self._lr + self._cov * (1 - self._lr)

        else:
            self._cov = cov_t
            self._mu = mu_t
            self._a = a_t
            self._b = b_t


    def _sample_posterior_predictive(self, x_t, n_samples=1):
        # 1. p(sigma^2)
        sigma_sq_t = invgamma.rvs(self._a, scale=self._b)

        try:
            # p(w|sigma^2) = N(mu, sigam^2 * cov)
            w_t = np.random.multivariate_normal(
                    self._mu, sigma_sq_t * self._cov
            )
        except np.linalg.LinAlgError as e:
            logger.debug("Error in {}".format(type(self).__name__))
            logger.debug('Errors: {}.'.format(e.args[0]))
            w_t = np.random.multivariate_normal(
                    np.zeros(self._d), np.eye(self._d)
            )

        # modify context
        u_t, S_t = x_t
        n_actions = S_t.shape[0]

        x_ta = [
                np.concatenate( (u_t, S_t[j, :], [1]) )
                for j in range(n_actions)
        ]
        assert len(x_ta[0]) == self._d

        # 2. p(r_new | params)
        mean_t_predictive = [
                np.dot(w_t, x_ta[j])
                for j in range(n_actions)
        ]

        cov_t_predictive = sigma_sq_t * np.eye(n_actions)
        r_t_estimates = np.random.multivariate_normal(
                            mean_t_predictive,
                            cov=cov_t_predictive, size=1
                        )
        r_t_estimates = r_t_estimates.squeeze()

        assert r_t_estimates.shape[0] == n_actions

        return r_t_estimates


    def choose_action(self, x_t):
        # p(R_new | params_t)
        r_t_estimates = self._sample_posterior_predictive(x_t)
        act = np.argmax(r_t_estimates)

        self._t += 1

        return act


    def update(self, a_t, x_t, r_t):
        u_t, S_t = x_t
        x_t = np.concatenate( (u_t, S_t[a_t, :], [1]) )

        # add a new data point
        if self._X_t is None:
            X_t = x_t[None, :]
            r_t_list = np.array([r_t])
        else:
            X_t, r_t_list = self._train_data[a_t]
            n = X_t.shape[0]
            X_t = np.vstack( (X_t, x_t))
            assert X_t.shape[0] == (n+1)
            assert X_t.shape[1] == self._d
            r_t_list = np.append(r_t_list, r_t)

        # train on a random batch
        n_samples = X_t.shape[0]
        if self._batch_mode and self._batch_size < n_samples:
            indices = np.arange(self._batch_size)
            batch_indices = np.random.choice(indices,
                                             size=self._batch_size,
                                             replace=False)
            X_t = X_t[batch_indices, :]
            r_t_list = r_t_list[batch_indices]

        self._train_data = (X_t, r_t_list)

        # exploration phase to mitigate cold-start
        # quite ill-defined
        if self._t < self._train_starts_at:
            return

        # update posterior prioridically
        # for computational reasons
        n_samples = X_t.shape[0]
        if n_samples % self._update_freq == 0:
            self._update_posterior(X_t, r_t_list)


class FeedForwardNetwork(nn.Module):
    """A fully connected neural network that shares parameters
    across all actions.

    Basically, solves a regression task on E[r_t | x_ta].

    For Robustness, adding a gradient noise is possible as per [1].

    [1]: https://arxiv.org/abs/1511.06807
    """
    def __init__(self, input_dim,
                       hidden_dim,
                       output_dim,
                       n_layer,
                       learning_rate,
                       set_gpu,
                       grad_noise,
                       gamma,
                       eta,
                       grad_clip,
                       grad_clip_norm,
                       grad_clip_value,
                       weight_decay,
                       debug):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential()
        self.model.add_module("input", nn.Linear(input_dim, hidden_dim))
        for i in range(n_layer - 1):
            self.model.add_module("fc{}".format(i+1), nn.Linear(hidden_dim, hidden_dim))
            self.model.add_module("relu{}".format(i+1), nn.ReLU())
        self.model.add_module("fc{}".format(n_layer), nn.Linear(hidden_dim, output_dim))

        self.set_gpu = set_gpu

        if set_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.model.cuda()
        else:
            self.device = torch.device("cpu")

        self.debug = debug

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

        self.opt = torch.optim.SGD(self.model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=weight_decay)

        # output layer implicitly defined by this
        self.criterion = nn.MSELoss()

        self._step = 0
        self.grad_noise = grad_noise
        self._gamma = gamma
        self._eta = eta

        if self.grad_noise:
            for m in self.model.modules():
                classname = m.__class__.__name__
                if classname.find("Linear") != -1:
                    m.register_backward_hook(self.add_grad_noise)

        self._grad_clip = grad_clip
        self._grad_clip_norm = grad_clip_norm
        self._grad_clip_value = grad_clip_value


    def predict(self, x):
        self.model.eval()
        data = Variable(x).to(self.device)
        return self.model(data)


    def train(self, epoch, train_loader):
        self.model.train()
        total_loss = 0.0

        # sample batches
        batches = train_loader.sample_batches()

        for batch_idx, (data, target) in enumerate(batches):


            # @todo: refactor this
            # type cast to match the default dtype of torch
            data = torch.from_numpy(data)
            data = data.float()
            target = torch.from_numpy(target)
            target = target.float()

            data = Variable(data).to(self.device)
            target = Variable(target).to(self.device)

            data = data.view(-1, self.input_dim)

            self.opt.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()

            if self._grad_clip:
                clip_grad_norm(self.model.parameters(), self._grad_clip_value, self._grad_clip_norm)

            if self.debug and batch_idx % 10 == 0:
                for layer in self.model.modules():
                    if np.random.rand() > 0.95:
                        if isinstance(layer, nn.Linear):
                            weight = layer.weight.data.numpy()
                            logger.debug("========================")
                            logger.debug("weight\n")
                            logger.debug("max:{}\tmin:{}\tavg:{}\n".format(weight.max(), weight.min(), weight.mean()))
                            grad = layer.weight.grad.data.numpy()
                            logger.debug("grad\n")
                            logger.debug("max:{}\tmin:{}\tavg:{}\n".format(grad.max(), grad.min(), grad.mean()))
                            logger.debug("=========================")

            self.opt.step()
            self._step += 1

            if np.isnan(loss.data.item()):
                raise Exception("gradient exploded or vanished: try clipping gradient")

            if batch_idx % 5 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rTrain Epoch: {:<2} [{:<5}/{:<5} ({:<2.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data.item()))

            total_loss += loss.data.item()

        sys.stdout.flush()

        return total_loss/len(train_loader)


    def _compute_grad_noise(self, grad):
        std = np.sqrt(self._eta / (1 + self._step)**self._gamma)
        return Variable(grad.data.new(grad.size()).normal_(0, std=std))


    def add_grad_noise(self, module, grad_i_t, grad_o):
        _, _, grad_i = grad_i_t[0], grad_i_t[1], grad_i_t[2]
        noise = self._compute_grad_noise(grad_i)
        return (grad_i_t[0], grad_i_t[1], grad_i + noise)


class SharedNeuralPolicy(object):
    """Policy that access a neural network reward estimator.

    For exploration, the epsilon greedy logic is set.

    """
    def __init__(self,
                 model,
                 dataloader,
                 train_starts_at=500,
                 train_freq=50,
                 set_gpu=False,
                 eps=0.1,
                 eps_anneal_factor=1e-5):
        self._model = model
        self._dataloader = dataloader
        self._train_starts_at = train_starts_at
        self._train_freq = train_freq
        self._t = 0
        self._t_update = 0
        self._set_gpu = set_gpu

        self._eps = eps
        self._eps_anneal_factor = eps_anneal_factor


    def choose_action(self, x_t):
        # make x_t -> x_ta
        u_t, S_t = x_t
        n_actions = S_t.shape[0]

        # torch.nn.Linear sets bias automatically
        # j x |u_t|
        U_t = np.tile(u_t, (n_actions, 1))
        # j x (|u_t| + |s_t|)
        X_ta = np.hstack( (U_t, S_t) )
        X_ta = torch.from_numpy(X_ta)
        # type cast to match the default dtype of torch
        X_ta = X_ta.float()
        # predict
        r_preds = self._model.predict(X_ta)

        if self._set_gpu:
            r_preds = r_preds.data.cpu().numpy().squeeze()
        else:
            r_preds = r_preds.data.numpy().squeeze()

        # eps-greedy
        u = np.random.uniform()
        if u > self._eps:
            a_t = np.argmax(r_preds)
        else:
            # choose random
            a_t = np.random.choice(np.arange(n_actions))

        # anneal eps
        self._eps *= (1 - self._eps_anneal_factor)**self._t

        return a_t


    def update(self, a_t, x_t, r_t):
        u_t, S_t = x_t
        x_ta = np.concatenate( (u_t, S_t[a_t, :]) )

        self._dataloader.add_sample(x_ta, r_t)
        self._t += 1

        # exploration phase to mitigate cold-start
        # quite ill-defined
        if self._t < self._train_starts_at:
            return

        # update posterior prioridically
        # for computational reasons
        n_samples = self._dataloader.n_samples
        if n_samples % self._train_freq == 0:
            self._t_update += 1
            # train one epoch (full data)?
            self._model.train(self._t_update, self._dataloader)
