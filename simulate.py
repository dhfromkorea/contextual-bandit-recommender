"""
given data, policy

simulate 1...T steps of contextual bandits

"""
import numpy as np

def simulate_contextual_bandit(data, policies=[]):
    """
    data: tuple
         a T-length sequence of contexts
    policy: list of policies

    returns:
        results: cumulative rewards, cumulative regret, simple regret, simple reward
        diagnostics
    """
    # infer T
    T = data[0].shape[0]

    results = {}

    for i, policy in enumerate(policies):

        rewards = np.zeros(T)
        regrets = np.zeros(T)
        t = 0

        # can regret be negative?
        for c_t, r_eat_t, r_no_eat_t, a_opt, is_poisonous in zip(*data):
            a_t = policy.choose_action(c_t)
            r_t = a_t * r_eat_t + (1 - a_t) * r_no_eat_t

            policy.update(a_t, c_t, r_t)

            r_t_opt =  a_opt * r_eat_t + (1 - a_opt) * r_no_eat_t

            rewards[t] = r_t
            regrets[t] = r_t_opt - r_t

            t += 1

        results[i] = {
                "policy": policy,
                "regrets": regrets,
                "rewards": rewards,
                "cum_rewards": np.sum(rewards),
                "simple_rewards": np.mean(rewards[-500:]),
                "cum_regret": np.sum(regrets),
                "simple_regret": np.mean(regrets[-500:])
                }

        #results[i] = (np.mean(rewards), np.mean(regrets))
    return results

