"""
"""

import sys

import numpy as np
from scipy.stats import invgamma

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import logging
logger = logging.getLogger(__name__ + "action_context_based_policy")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



class SharedLinUCBPolicy(object):
    """
    action-context aware policy

    unlike LinUCB (disjoint) LinUCB (hybrid) in [1],
    all actions share all parameters.

    expected to underperform in a high data regime.
    expected to perform relatively well in a low data regime.

    Just a baseline.

    I did not like the idea of building a disjoint model
    for each action.

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

        #self._theta = np.linalg.lstsq(self._A, self._b, rcond=None)[0]
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
        """

        solve X_a w_a + b_a = r_a

        where
        X_a in R^{n x d}
        b_a in R^{n x 1}
        D_a (n_j x d): design matrix for a_j
        theta (d x 1)

        solve D_a theta_a = c_a
        theta_a = (D_a^TD_a + I_d)^{-1}
        A_a = D_a^TD_a + I_d

        I_d perturbation singular
        X_a (d x d):

        alpha: constant coefficient for ucb
        at least 1 - delta probability
        alpha = 1 + sqrt(ln(2/delta)/2)
        copmute ucb

        assumes alpha given

        access to theta_a for all actions
        """
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

        # todo: tiebreaking
        a_t = np.argmax(Q)

        #if self._t % 10000 == 0:
        #    logger.debug("Q est {}".format(Q))
        #    logger.debug("ubc {} at {}".format(ubc_t[a_t], self._t))

        self._t += 1

        return a_t

    def update(self, a_t, x_t, r_t):
        """
        """

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
            # solve linear systems for one action
            # using lstsq to handle over/under determined systems
            #self._theta = np.linalg.lstsq(self._A, self._b, rcond=None)[0]
            self._A_inv = np.linalg.inv(self._A)
            self._theta = self._A_inv.dot(self._b)


class SharedLinearGaussianThompsonSamplingPolicy(object):
    """

    action-context aware policy

    A variant of Thompson Sampling that is based on

    Bayesian Linear Regression

    with Inverse Gamma and Gaussian prior


    unlike the one in context_based_policy.py
    all parameters are shared across actions

    the performance expectation is similar as SharedLinUCBPolicy.

    """

    def __init__(self,
                 context_dim,
                 eta_prior=6.0,
                 lambda_prior=0.25,
                 train_starts_at=500,
                 posterior_update_freq=50,
                 batch_mode=True,
                 batch_size=256,
                 lr=0.01
    ):
        """
        a_0; location for IG t=0
        b_0; scale for IG t=0


        """

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
        """
        p(w, sigma^2) = p(mu|cov)p(a, b)

        where p(sigma^2) ~ Inverse Gamma(a_0, b_0)
              p(w|sigma) ~ N(mu_0, sigma^2 * lambda_0^-1)

        does a full refit. online version could be implemented.

        """
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
        """

        estimate

        p(R_new | X, R_old)
        = int p(R_new | params )p(params| X, R_old) d theta

        """

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
        """
        @todo: unify c_t vs. x_t

        with respect to a_t
        """

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
    """
    a simple feedforward network
    that estimates E[R_t | X_ta]

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
                #nn.init.constant_(m.bias, 0)

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
        """TODO: Docstring for train.
        Parameters
        ----------
        arg1 : TODO
        Returns
        -------
        TODO
        """
        self.model.train()
        total_loss = 0.0

        # sample batches
        batches = train_loader.sample()

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
                clip_grad_norm_(self.model.parameters(), self._grad_clip_value, self._grad_clip_norm)


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
        print("")

        return total_loss/len(train_loader)


    def _compute_grad_noise(self, grad):
        """TODO: Docstring for function.
        Parameters
        ----------
        arg1 : TODO
        Returns
        -------
        TODO
        """

        std = np.sqrt(self._eta / (1 + self._step)**self._gamma)
        return Variable(grad.data.new(grad.size()).normal_(0, std=std))


    def add_grad_noise(self, module, grad_i_t, grad_o):
        """TODO: Docstring for add_noise.
        Parameters
        ----------
        arg1 : TODO
        Returns
        -------
        TODO
        """
        _, _, grad_i = grad_i_t[0], grad_i_t[1], grad_i_t[2]
        noise = self._compute_grad_noise(grad_i)
        return (grad_i_t[0], grad_i_t[1], grad_i + noise)


class NeuralPolicy(object):
    def __init__(self,
                 model,
                 dataloader,
                 train_starts_at=500,
                 train_freq=50,
                 set_gpu=False
        ):
        self._model = model
        self._dataloader = dataloader
        self._train_starts_at = train_starts_at
        self._train_freq = train_freq
        self._t = 0
        self._t_update = 0
        self._set_gpu = set_gpu


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

        #r_preds = np.zeros(n_actions)
        #for j in range(n_actions):
        #    x_ta = np.concatenate( (u_t, S_t[j, :]) )
        #    x_ta = torch.from_numpy(x_ta)
        #    # type cast to match the default dtype of torch
        #    x_ta = x_ta.float()
        #    # predict
        #    r_pred = self._model.predict(x_ta)
        #    if self._set_gpu:
        #        r_preds[j] = r_pred.data.cpu().numpy()[0]
        #    else:
        #        r_preds[j] = r_pred.data.numpy()[0]


        # @todo: consider proper tie-breaking
        # encourage exploration
        np.random.seed(0)
        noise = np.random.uniform(0, 0.25, size=len(r_preds))
        a_t = np.argmax(r_preds + noise)

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


