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

        for x_t, r_acts, a_t_opt, _ in zip(*data):
            a_t = policy.choose_action(x_t)
            r_t = r_acts[a_t]

            policy.update(a_t, x_t, r_t)

            r_t_opt = r_acts[a_t_opt]

            regret_t = r_t_opt - r_t

            results[i]["log"][:, t] = [a_t, a_t_opt, r_t, regret_t]

            t += 1

        results[i]["policy"] = policy
        regrets = results[i]["log"][3, :]
        results[i]["cum_regret"] = np.cumsum(regrets)
        results[i]["simple_regret"] = np.sum(regrets[-500:])

    return results


def simulate_contextual_bandit_partial_label(data_generator, n_samples, policies):
    """
    data: generator
    """
    # infer T
    results = [None] * len(policies)
    for i, policy in enumerate(policies):
        results[i] = {}
        results[i]["reward"] = []

        t = 0
        t_1 = 0
        for uv in data_generator:
            # s_t = a list of article features
            u_t, S_t, r_acts, act_hidden = uv

            x_t = (u_t, S_t)
            a_t = policy.choose_action(x_t)
            if a_t == act_hidden:
                assert act_hidden == r_acts[0]
                # useful
                r_t = r_acts[1]
                policy.update(a_t, x_t, r_t)
                results[i]["reward"].append(r_t)
                t_1 += 1
            else:
                # not useful
                # for off-policy learning
                pass

            #if t % 100000 == 0:
            #    print("training with {} samples so far".format(t_1))

            t += 1

            print("")
            if t_1 > n_samples:
                print("{:.2f}% data useful".format(t_1/t * 100))
                break


        results[i]["policy"] = policy
        rewards = results[i]["reward"]
        results[i]["cum_reward"] = np.cumsum(rewards)
        results[i]["CTR"] = np.array(rewards)

    return results


