import os
import json
from random import random, seed

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

from deephyper.search import util
from deephyper.search.nas import NeuralArchitectureSearch
from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.evaluator.evaluate import Encoder

# from deephyper.search.nas.env.neural_architecture_envs import NeuralArchitectureVecEnv
from nascd.env.nasenv import NasEnv2

ENV = NasEnv2

dhlogger = util.conf_logger("nascd.search.rtg_pg")


class RtgPG(NeuralArchitectureSearch):
    """Search class to run a full random neural architecture search. The search is filling every available nodes as soon as they are detected. The master job is using only 1 MPI rank.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.search.nas.model.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
    """

    def __init__(self, problem, run, evaluator, **kwargs):

        super().__init__(problem=problem, run=run, evaluator=evaluator, **kwargs)

        seed(self.problem.seed)

        if evaluator == "balsam":
            balsam_launcher_nodes = int(os.environ.get("BALSAM_LAUNCHER_NODES", 1))
            deephyper_workers_per_node = int(
                os.environ.get("DEEPHYPER_WORKERS_PER_NODE", 1)
            )
            n_free_nodes = balsam_launcher_nodes - 1  # Number of free nodes
            self.free_workers = (
                n_free_nodes * deephyper_workers_per_node
            )  # Number of free workers
        else:
            self.free_workers = 1

        dhlogger.info(
            jm(
                type="start_infos",
                alg="rtg_pg",
                nworkers=self.free_workers,
                encoded_space=json.dumps(self.problem.space, cls=Encoder),
            )
        )

    @staticmethod
    def _extend_parser(parser):
        NeuralArchitectureSearch._extend_parser(parser)
        return parser

    def main(self):

        # Setup
        num_envs = 1
        N = 8
        env = build_env(num_envs, self.problem, self.evaluator)
        batch_size = N * env.num_actions_per_env  # TODO
        epochs = 300
        lr = 1e-2
        train(env, lr=lr, epochs=epochs, batch_size=batch_size)


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class PolicyGRU(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        self.layers = nn.ModuleList()
        # >>> rnn = nn.GRU(10, 20, 2)
        # >>> input = torch.randn(5, 3, 10)
        # >>> h0 = torch.randn(2, 3, 20)
        # >>> output, hn = rnn(input, h0)
        # print("sizes: ", sizes)
        nh = 32  # number of units in gru cell
        nl = 1  # number of layers
        self.rnn = nn.GRU(sizes[0], nh, nl)
        self.h0 = torch.randn(nl, 1, nh)
        self.layer_out = nn.Linear(nh, sizes[-1])
        # for j in range(len(sizes) - 1):
        #     act = activation if j < len(sizes) - 2 else output_activation
        #     self.layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        #     self.layers.append(act())
        self.hn = self.h0

    def forward(self, x, done=False):
        if done:
            self.hn = self.h0
        if len(x.size()) == 3:
            output, self.hn = self.rnn(x, self.h0)
        else:
            x = x.unsqueeze(0)
            # print("x.size: ", x.size(), len(x.size()))
            output, self.hn = self.rnn(x, self.hn)
            output = output.squeeze(0)
        # print("output: ", output)
        output = self.layer_out(output)

        return output


class PolicyLSTM(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        self.layers = nn.ModuleList()
        # >>> rnn = nn.LSTM(10, 20, 2)
        # >>> input = torch.randn(5, 3, 10)
        # >>> h0 = torch.randn(2, 3, 20)
        # >>> c0 = torch.randn(2, 3, 20)
        # >>> output, (hn, cn) = rnn(input, (h0, c0))
        nh = 32  # number of units in gru cell
        nl = 1  # number of layers
        self.rnn = nn.LSTM(sizes[0], nh, nl)
        self.h0 = torch.randn(nl, 1, nh)
        self.c0 = torch.randn(nl, 1, nh)
        self.layer_out = nn.Linear(nh, sizes[-1])
        self.hn = self.h0
        self.cn = self.c0

    def forward(self, x, done=False):
        if done:
            self.hn = self.h0
            self.cn = self.c0
        if len(x.size()) == 3:
            output, (self.hn, self.cn) = self.rnn(x, (self.h0, self.c0))
        else:
            x = x.unsqueeze(0)
            output, (self.hn, self.cn) = self.rnn(x, (self.hn, self.cn))
            output = output.squeeze(0)
        # print("output: ", output)
        output = self.layer_out(output)

        return output


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def train(env, hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    # env = build_env(num_envs, problem, evaluator)
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, Discrete
    ), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]  # TODO
    print(f"obs_dim: {obs_dim}")
    n_acts = env.action_space.n

    # make core of policy network
    # logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    # logits_net = PolicyLSTM(sizes=[obs_dim, n_acts])
    logits_net = PolicyGRU(sizes=[obs_dim, n_acts])

    # make function to compute action distribution
    def get_policy(obs, done):
        logits = logits_net(obs, done)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs, done):
        return get_policy(obs, done).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs, True).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for reward-to-go weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())
            # print(batch_obs)

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32), done)
            obs, rew, done, _ = env.step([act])
            # print("rew: ", rew)
            # print("obs: ", obs)

            # save action, reward
            batch_acts.append(act)  # x
            ep_rews.append(rew)  # y

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                # batch_weights += list(reward_to_go(ep_rews))
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                # print(f"len(batch_ops:{len(batch_obs)}, batch_size:{batch_size}")
                # print("len(batch_obs): ", len(batch_obs))
                if len(batch_obs) == batch_size:
                    # exit()
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )


def build_env(num_envs, problem, evaluator):
    """Build nas environment.

    Args:
        num_envs (int): number of environments to run in parallel (>=1).
        space (dict): space of the search (i.e. params dict)
        evaluator (Evaluator): evaluator object to use.

    Returns:
        VecEnv: vectorized environment.
    """
    assert num_envs >= 1, f"num_envs={num_envs}"
    space = problem.space
    cs_kwargs = space["create_search_space"].get("kwargs")
    if cs_kwargs is None:
        search_space = space["create_search_space"]["func"]()
    else:
        search_space = space["create_search_space"]["func"](**cs_kwargs)
    env = ENV(num_envs, space, evaluator, search_space)
    return env


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env_name", "--env", type=str, default="CartPole-v0")
#     parser.add_argument("--render", action="store_true")
#     parser.add_argument("--lr", type=float, default=1e-2)
#     args = parser.parse_args()
#     print("\nUsing reward-to-go formulation of policy gradient.\n")
#     train(env_name=args.env_name, render=args.render, lr=args.lr)

if __name__ == "__main__":
    args = RtgPG.parse_args()
    search = RtgPG(**vars(args))
    search.main()
