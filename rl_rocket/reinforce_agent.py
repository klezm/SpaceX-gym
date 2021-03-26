import pathlib
import time
import collections
import json
# import sys
from pprint import pprint
from types import SimpleNamespace
from typing import Optional
from os import PathLike

import imageio
import PIL.Image
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

# ML stuff
import gym
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# import jax.numpy as jnp
# from jax import grad, jit, vmap

# import huskarl as hk # https://github.com/danaugrs/huskarl

# import ray
# from ray import tune

# import stable_baselines3 as baselines

import rl_rocket.constants as CONST
from rl_rocket.policies import PolicyEstimatorNet


class REINFORCEAgent:
    def __init__(self,
                 env: gym.Env,
                 # state_dim,
                 # action_dim,
                 t_max,
                 tb_log: SummaryWriter,
                 alpha = 1e-3,
                 alpha_min = 1e-6,
                 gamma = .995,
                 epsilon = .1,
                 net = None,
                 video = None,
                 curriculum = True,
                 hidden_layer_size = 100,
                 n_hidden = 1,
                 save_path: Optional[PathLike] = True
                 ):
        """
        Implementation of REINFORCE algorithm.

        :param state_dim:
        :param action_dim:
        :param t_max:
        :param tb_log: summary writer for tensorboard
        :param alpha:
        :param gamma:
        :param epsilon:
        :param tb_log: summary writer for tensorboard
        :param net: optionally provide a nn.Sequential(...) network to be used ofr the policy
        """
        # super().__init__(*args, **kwargs)
        self.state_dim = env.observation_space.high.size
        self.action_dim = env.action_space.high.size
        self.t_max = t_max
        self.α = alpha
        # self.α_this = self.α
        self.α_min = alpha_min
        self.γ = gamma
        self.ɛ = epsilon
        self.G_baseline = 0  # Mean of the so far seen G's
        # self.policy_net: nn.Model = None
        self.neg_log_likelihood = nn.GaussianNLLLoss()  # negative log likelihood
        self.tb_log = tb_log
        self.env = env
        self.video = video
        self.curriculum = self.env.C.CURRICULUM

        # create storage to keep track of the states, actions and rewards in one episode
        self.reset()
        self.total_steps = 0
        self.episode_i = 0
        self.reward_hist = collections.deque(maxlen = CONST.MEAN_REWARD_LEN)
        self._mean_reward = None
        self.mean_reward_best = None
        self.mean_reward_best_id = None
        self.save_path: str = (
                          # pathlib.Path(f'/content/save/{time.strftime("%y-%m-%d_%H-%M-%S", time.gmtime())}')
                          # if save_path is True and "COLAB_GPU" in os.environ else
                          pathlib.Path(f'save/{time.strftime("%y-%m-%d_%H-%M-%S", time.gmtime())}')
                          if save_path is True else save_path
                          # if save_path is True else pathlib.Path(save_path)
                          # if save_path else save_path
        )

        # setting up the policy estimating network
        self.policy_net = PolicyEstimatorNet(self.state_dim, self.action_dim, net = net,
                                             hidden_size = hidden_layer_size, n_hidden = n_hidden)
        # print(self.policy_net)
        self.tb_log.add_graph(model = self.policy_net, input_to_model = torch.Tensor(self.env.reset()))
        self.tb_log.add_text("hparam/constants", json.dumps(vars(self.env.C), indent = 2), global_step = 0)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), self.α)

        # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
        # https://www.programmersought.com/article/12164650026/
        # https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 50, gamma = self.γ)
        # https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no/48324389#48324389
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.γ)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = self.lr_step)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, )
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr = self.α_min, max_lr = self.α, step_size_up = 25_000)

    @property
    def mean_reward(self):
        return np.mean(self.reward_hist)

    @mean_reward.setter
    def mean_reward(self, val):
        self._mean_reward = val

    @property
    def save_path(self):
        if self._save_path is None:
            return None
        return pathlib.Path(self._save_path)

    @save_path.setter
    def save_path(self, value):
        if value is None:
            self._save_path = value
        else:
            # self._save_path = str(value)
            self._save_path = pathlib.Path(value).as_posix()

    # def lr_step(self, episode):
    #     # print("episode:", episode)  # 0, 1, 2, ...
    #     # α = self.γ ** episode
    #     # α = self.α * np.exp(-self.γ * self.episode_i)
    #     # α = self.scheduler.get_last_lr()[0]
    #     # if self.γ * α > self.α_min:
    #     if self.α_this > self.α_min:
    #         # print(self.α_this > self.α_min, self.α_this, self.α_min)
    #         self.α_this *= self.γ * episode
    #         return self.γ ** episode
    #         # self.γ ** episode
    #     else:
    #         # self.α_this = episode
    #         # return self.γ ** self.α_this
    #         return self.α_min / self.α_this
    #         # return 1
    #         # return self.α_min

    def reset(self):
        """
        Resets the current episode
        """
        self.T = 0
        self.s = torch.empty((self.t_max, self.state_dim))
        self.a = torch.empty(self.t_max, self.action_dim)
        self.r = torch.empty(self.t_max)

    def policy(self, state):
        """
        Predicts an action on a given state.
        """
        mean, std = self.policy_net(state)
        e = torch.Tensor(np.random.normal(0, 1, self.action_dim))  # TODO: evtl detach()
        action = mean + std * e
        return action

    def get_nll(self, state, action):
        """
        Negative log likelihood
        """
        # get action distribution
        mean, std = self.policy_net(state)
        # std needs to be positive:
        std = torch.abs(std)
        # return log likelihood of sampled action
        return self.neg_log_likelihood(action, mean, std)

        # pdf_value = 1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-0.5 * ((action - mean) / std)**2)
        # # get log probability
        # log_prob = torch.log(pdf_value)

    def step(self, state, reward, done):
        # print("s", self.s)
        # print("T", self.T)
        # print("state", state)
        self.s[self.T] = state
        self.r[self.T] = reward

        if self.video:
            self.video.append_data(self.env.render(mode = "rgb_array"))

        if done:
            #  training
            loss = 0
            G = np.zeros(self.T)
            # TODO: performance
            for t in range(self.T):
                disc_reward = 0
                for t2 in range(t, self.T):
                    disc_reward += self.γ**(t2 - t) * self.r[t2 + 1]
                G[t] += disc_reward
                loss += (G[t] - self.G_baseline) * self.get_nll(self.s[t], self.a[t])  # get_nll returns the negative log likelihood for pi(a | s)
                self.G_baseline += 1 / self.total_steps * (G[t] - self.G_baseline)

            # Zero out all of the gradients for the variables which the optimizer will update.
            self.optimizer.zero_grad()
            # This is the backwards pass: compute the gradient of the loss with respect to each  parameter of the model.
            loss.backward()
            # Actually update the parameters of the model using the gradients computed by the backwards pass.
            self.optimizer.step()
            if self.scheduler.get_last_lr()[0] > self.α_min:  # if the learning rate is above α_min then do decay
                self.scheduler.step()
            # self.α_this = self.scheduler.get_last_lr()[0]

            # print(f"Episode took {self.T} steps, and gave a total loss of {loss}")
            self.tb_log.add_scalar("training/policy_loss", loss, global_step = self.episode_i)
            self.tb_log.add_scalar("training/episode_length", self.T, global_step = self.total_steps)
            # self.reset()
        else:
            action = self.policy(state)
            self.a[self.T] = action
            self.T += 1
            return action

    def train(self, total_episodes = CONST.MAX_EPISODE):
        for i in range(total_episodes):
            self.episode_i = i

            if self.curriculum:
                # # self.env.close()
                # # self.env = GymRocketLander(constants = curriculum_decay(self.episode_i))
                # self.env.C = curriculum_decay(self.episode_i)
                # # c = curriculum_decay(self.episode_i)
                # # self.env.C.init(INIT_X = c.INIT_X, INIT_Y = c.INIT_Y)
                # self.env.reset()
                self.tb_log.add_scalar("env/init_height", self.env.C.INIT_Y, global_step = self.episode_i)
            self.reset()
            state = torch.Tensor(self.env.reset())
            reward_episode = 0
            reward, done = 0, False
            while True:
                self.total_steps += 1
                action = self.step(state, reward, done)
                if done:
                    # state, = self.env.step(np.array(action.detach()))  # Dieser state wird nie verwendet. Macht das Sinn?
                    # print(f"Final Reward of episode: {reward}")
                    break
                state, reward, done, _ = self.env.step(np.array(action.detach()))
                state = torch.tensor(state.astype(np.float32))
                reward_episode += reward
                self.reward_hist.append(reward_episode)

                self.tb_log.add_scalar("step/reward", reward, global_step = self.total_steps)
                if self.T + 1 >= self.t_max:  # Interupt episode when maximum of timesteps is reached
                    print(f"time expired after {self.T} steps")
                    # reward = -10  # TODO: should we give a big negative reward for timeouts?
                    done = True
                    # break
            self.tb_log.add_scalar("training/reward", reward_episode, global_step = self.episode_i)
            self.tb_log.add_scalar("training/α", self.scheduler.get_last_lr()[0], global_step = self.episode_i)
            # self.tb_log.add_scalar("training/α", self.optimizer.param_groups[0]["lr"], global_step = self.episode_i)
            self.tb_log.add_scalar(f'training/mean_reward_{CONST.MEAN_REWARD_LEN}', np.mean(self.reward_hist), global_step = self.episode_i)
            FIRST_CHECKPOINT = 100
            if ((
                    self.mean_reward_best_id is not None
                    and self.mean_reward > self.mean_reward_best
                    # and FIRST_CHECKPOINT < self.episode_i
                    # and FIRST_CHECKPOINT < self.mean_reward_best_id <= self.episode_i + 10
                )
                or self.episode_i == FIRST_CHECKPOINT
            ):
                self.save()
                self.mean_reward_best_id = self.episode_i
                self.mean_reward_best = self.mean_reward
                # self.save(self.save_path.joinpath(f'{self.episode_i:{len(str(CONST.MAX_EPISODE))}}'))
        # self.tb_log.add_hparams(vars(self.env.C), {f'hparam/last_{CONST.MEAN_REWARD_LEN}_mean_reward': np.mean(self.reward_hist)})
        self.tb_log.close()
        self.env.close()

    def play(self, n_episodes = 1):
        # self.policy_net.eval()
        for n_episode in range(n_episodes):
            self.reset()
            state = torch.Tensor(self.env.reset())
            reward_episode = 0
            reward, done = 0, False
            with torch.no_grad():
                while True:
                    self.total_steps += 1
                    if done:
                        break
                    action = self.step(state, reward, done)
                    state, reward, done, _ = self.env.step(np.array(action.detach()))
                    state = torch.tensor(state.astype(np.float32))
                    reward_episode += reward
                    self.reward_hist.append(reward_episode)

                    if self.T + 1 >= self.t_max:  # Interupt episode when maximum of timesteps is reached
                        print(f"time expired after {self.T} steps")
                        done = True
                        break
                self.env.close()

    def save(self, save_path = None):
        if save_path is None:
            save_path = self.save_path
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents = True, exist_ok = True)

        def type_converter(data):
            if isinstance(data, collections.deque):
                return list(data)
            else:
                return data

        # torch.save(self.policy_net.state_dict(), save_path.joinpath(f'model_{self.episode_i:0{len(str(CONST.MAX_EPISODE))}}.pth'))
        torch.save(self.policy_net.state_dict(), save_path.joinpath(f'model_{self.episode_i}.pth'))

        data = {
            "agent": {k: type_converter(v) for k, v in vars(self).items() if
                      not isinstance(v, (torch.Tensor))
                      and k not in ["env", "optimizer", "scheduler", "scheduler", "tb_log", "policy_net", "neg_log_likelihood"]
                      # and not k.startswith("_")
                      },
            "C": vars(self.env.C),
            "env": {k: v for k, v in vars(self.env).items() if
                    not isinstance(v, gym.Space)
                    and k not in ["world", "containers", "water", "drawlist", "ship", "lander", "legs", "C", "np_random"]},
         }

        # data["agent"].update({"mean_reward": self.mean_reward})

        with open(save_path.joinpath("agent.json"), "w+") as f:
            json.dump(data, f, indent = 2)

    def load(self, folder):
        folder = pathlib.Path(folder)
        # model = sorted([x for x in folder.glob("model*.pth")])[-1]
        model = sorted([x for x in folder.glob("model*.pth")], key = lambda x: int(x.stem.split("_")[-1]))[-1]
        self.policy_net.load_state_dict(torch.load(model))
        # self.policy_net.eval()

        with open(folder.joinpath("agent.json")) as f:
            json_agent = json.load(f)

        for k, v in json_agent["agent"].items():
            # if k == "_mean_reward":
            #     setattr(self, "mean_reward", v)
            if k == "reward_hist":
                setattr(self, k, collections.deque(v, maxlen = CONST.MEAN_REWARD_LEN))
            else:
                setattr(self, k, v)
        for k, v in json_agent["C"].items():
            setattr(self.env.C, k, v)
        # self.env.C.__init(**json.load(folder.joinpath(self._save_pths["C"])))
        for k, v in json_agent["env"].items():
            setattr(self.env, k, v)


if __name__ == "__main__":
    writer = SummaryWriter(
        # log_dir = CONST.LOG_DIR,
        # comment = f'{time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())}'
    )

    # env = gym.make("gym_rocketlander:rocketlander-v0")
    # env = gym.make('gym_goddard:Goddard-v0')
    # env = gym.make('gym_goddard:GoddardSaturnV-v0')
    # env = gym.make("CartPole-v1")
    # env = gym.make("LunarLanderContinuous-v2")
    from gyms.rocket_lander_env import C, GymRocketLander
    env = GymRocketLander(constants = C(SHAPING_REWARD = False,
                                        FUELCOST_REWARD = False,
                                        CURRICULUM = True,
                                        ))

    video = None
    # video = imageio.get_writer("test.mp4", fps = 30)

    agent = REINFORCEAgent(
        env = env,
        # state_dim = env.observation_space.shape,
        # state_dim = env.observation_space.high.size,
        # action_dim = env.action_space.high.size,
        t_max = CONST.MAX_EPISODE_LEN,
        tb_log = writer,
        alpha = 1e-3,
        alpha_min = 1e-4,
        # alpha = 1e-3,
        # gamma = .99,
        # epsilon = .1,
        video = video,
        curriculum = True,
    )
    # agent.train(total_episodes = 10)
    agent.train()

    if video is not None: video.close()
