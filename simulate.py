"""
given data, policy

simulate 1...T steps of contextual bandits

"""
import numpy as np

def simulate_contextual_bandit(data, n_samples, policies):
    """
    data: tuple
         a T-length sequence of contexts
    policy: list of policies

    returns:
        results: cumulative rewards, cumulative regret, simple regret, simple reward
        diagnostics


    (diag)
    - simple reward: 500 steps average
    - cum rewards: over T steps
    - action value estimates: snapshot every n steps (x axis: actions, y: value)

    (eval)
    - simple regret
    - cum regrets
    - regret: over time (peak)
    """
    # infer T
    results = [None] * len(policies)

    for i, policy in enumerate(policies):
        results[i] = {}
        # log a_t, r_t, del_t (regret)
        results[i]["log"] = np.zeros((4, n_samples))

        t = 0

        for c_t, r_acts, a_t_opt, _ in zip(*data):
            a_t = policy.choose_action(c_t)
            r_t = r_acts[a_t]

            policy.update(a_t, c_t, r_t)

            r_t_opt =  r_acts[a_t_opt]

            regret_t = r_t_opt - r_t

            results[i]["log"][:, t] = [a_t, a_t_opt, r_t, regret_t]

            t += 1

        results[i]["policy"] = policy
        regrets = results[i]["log"][3, :]
        results[i]["cum_regret"] = np.cumsum(regrets)
        results[i]["simple_regret"] = np.sum(regrets[-500:])

    return results

