from pprint import pprint
import gym


def print_env_info(env: gym.Env):
    print("\n" "--------------------     env     --------------------")
    pprint(vars(env))

    try:
        print("\n" "--------------------     env.env     --------------------")
        pprint(vars(env.env))
    except:
        pass

    print("\n" "--------------------     env.action_space     --------------------")
    pprint(vars(env.action_space))

    print("\n" "--------------------     env.observation_space     --------------------")
    pprint(vars(env.observation_space))

