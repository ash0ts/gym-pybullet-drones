"""Learning script for multi-agent problems.

Agents are based on `ray[rllib]`'s implementation of PPO and use a custom centralized critic.

Example
-------
To run the script, type in a terminal:

    $ python multiagent.py --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --num_workers <num_workers>

Notes
-----
Check Ray's status at:

    http://127.0.0.1:8265

"""
import os
import time
import argparse
from datetime import datetime
from sys import platform
import subprocess
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import ENV_STATE

from ray.air.integrations.wandb import WandbLoggerCallback  # 🪄🐝
import wandb

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger

import shared_constants
from utils import video_recordings_to_wandb_table

OWN_OBS_VEC_SIZE = None  # Modified at runtime
ACTION_VEC_SIZE = None  # Modified at runtime

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

############################################################


class CustomTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.action_model = FullyConnectedNetwork(
            Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE, )),
            action_space,
            num_outputs,
            model_config,
            name + "_action"
        )
        self.value_model = FullyConnectedNetwork(
            obs_space,
            action_space,
            1,
            model_config,
            name + "_vf"
        )
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        value_out, _ = self.value_model(
            {"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])

############################################################


class FillInActions(DefaultCallbacks):
    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            # Box(-np.inf, np.inf, (ACTION_VEC_SIZE,), np.float32) # Unbounded
            Box(-1, 1, (ACTION_VEC_SIZE,), np.float32)  # Bounded
        )
        _, opponent_batch = original_batches[other_id]
        # opponent_actions = np.array([action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]) # Unbounded
        opponent_actions = np.array([action_encoder.transform(
            np.clip(a, -1, 1)) for a in opponent_batch[SampleBatch.ACTIONS]])  # Bounded
        to_update[:, -ACTION_VEC_SIZE:] = opponent_actions

# class ModelEvaluationCallback(DefaultCallbacks):
#     # def __init__(self):
#         # self.logger =


############################################################
def central_critic_observer(agent_obs, **kw):
    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            # Filled in by FillInActions
            "opponent_action": np.zeros(ACTION_VEC_SIZE),
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            # Filled in by FillInActions
            "opponent_action": np.zeros(ACTION_VEC_SIZE),
        },
    }
    return new_obs


############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=5,                 type=int,
                        help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default='leaderfollower',  type=str,             choices=[
                        'leaderfollower', 'flock', 'meetup'],      help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs',         default='kin',             type=ObservationType,
                        help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',         default='one_d_rpm',       type=ActionType,
                        help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--algo',        default='cc',              type=str,             choices=[
                        'cc'],                                     help='MARL approach (default: cc)', metavar='')
    parser.add_argument('--workers',     default=0,                 type=int,
                        help='Number of RLlib workers (default: 0)', metavar='')
    parser.add_argument('--record',      default=True,              type=bool,
                        help='Save screenshots of training (default: True)', metavar='')
    ARGS = parser.parse_args()

    #### Save directory ########################################
    output_folder = PROJECT_NAME = "multiagent-drone-pybullet-rllib"
    exp = ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo + \
        '-'+ARGS.obs.value+'-'+ARGS.act.value

    filename = os.path.dirname(os.path.abspath(
        __file__))+f'/{output_folder}/results-'+exp+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    videos_folder = os.path.dirname(os.path.abspath(
        __file__))+f'/{output_folder}/videos-'+exp+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder+'/')

    eval_folder = os.path.dirname(os.path.abspath(
        __file__))+f'/{output_folder}/eval-'+exp+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder+'/')

    eval_videos = f"{eval_folder}/videos"
    if not os.path.exists(eval_videos):
        os.makedirs(eval_videos+'/')

    eval_logs = f"{eval_folder}/logs"
    if not os.path.exists(eval_logs):
        os.makedirs(eval_logs+'/')

    #### Print out current git commit hash #####################
    # if platform == "linux" or platform == "darwin":
    #     git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
    #     with open(filename+'/git_commit.txt', 'w+') as f:
    #         f.write(str(git_commit))

    #### Constants, and errors #################################
    if ARGS.obs == ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 12
    elif ARGS.obs == ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit()
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ARGS.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VEC_SIZE = 1
    elif ARGS.act in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VEC_SIZE = 4
    elif ARGS.act == ActionType.PID:
        ACTION_VEC_SIZE = 3
    else:
        print("[ERROR] unknown ActionType")
        exit()

    #### Uncomment to debug slurm scripts ######################
    # exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the custom centralized critic model ##########
    ModelCatalog.register_custom_model(
        "cc_model", CustomTorchCentralizedCriticModel)

    #### Register the environment ##############################
    temp_env_name = "this-aviary-v0"
    if ARGS.env == 'flock':
        register_env(temp_env_name, lambda _: FlockAviary(num_drones=ARGS.num_drones,
                                                          aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                          obs=ARGS.obs,
                                                          act=ARGS.act,
                                                          record=ARGS.record,
                                                          output_folder=videos_folder
                                                          )
                     )
    elif ARGS.env == 'leaderfollower':
        register_env(temp_env_name, lambda _: LeaderFollowerAviary(num_drones=ARGS.num_drones,
                                                                   aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                                   obs=ARGS.obs,
                                                                   act=ARGS.act,
                                                                   record=ARGS.record,
                                                                   output_folder=videos_folder
                                                                   )
                     )
    elif ARGS.env == 'meetup':
        register_env(temp_env_name, lambda _: MeetupAviary(num_drones=ARGS.num_drones,
                                                           aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                           obs=ARGS.obs,
                                                           act=ARGS.act,
                                                           record=ARGS.record,
                                                           output_folder=videos_folder
                                                           )
                     )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    #### Unused env to extract the act and obs spaces ##########
    if ARGS.env == 'flock':
        temp_env = FlockAviary(num_drones=ARGS.num_drones,
                               aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                               obs=ARGS.obs,
                               act=ARGS.act,
                               record=ARGS.record,
                               output_folder=videos_folder
                               )
    elif ARGS.env == 'leaderfollower':
        temp_env = LeaderFollowerAviary(num_drones=ARGS.num_drones,
                                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                        obs=ARGS.obs,
                                        act=ARGS.act,
                                        record=ARGS.record,
                                        output_folder=videos_folder
                                        )
    elif ARGS.env == 'meetup':
        temp_env = MeetupAviary(num_drones=ARGS.num_drones,
                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                obs=ARGS.obs,
                                act=ARGS.act,
                                record=ARGS.record,
                                output_folder=videos_folder
                                )
    else:
        print("[ERROR] environment not yet implemented")
        exit()
    observer_space = Dict({
        "own_obs": temp_env.observation_space[0],
        "opponent_obs": temp_env.observation_space[0],
        "opponent_action": temp_env.action_space[0],
    })
    action_space = temp_env.action_space[0]

    #### Note ##################################################
    # RLlib will create ``num_workers + 1`` copies of the
    # environment since one copy is needed for the driver process.
    # To avoid paying the extra overhead of the driver copy,
    # which is needed to access the env's action and observation spaces,
    # you can defer environment initialization until ``reset()`` is called

    #### Set up the trainer's config ###########################
    # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = ppo.DEFAULT_CONFIG.copy()
    config = {
        "env": temp_env_name,
        # "num_workers": 0 + ARGS.workers,
        "num_workers": 3,
        "num_gpus": 1,  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "callbacks": FillInActions,
        "framework": "torch",
    }

    #### Set up the model parameters of the trainer's config ###
    config["model"] = {
        "custom_model": "cc_model",
    }

    #### Set up the multiagent params of the trainer's config ##
    config["multiagent"] = {
        "policies": {
            "pol0": (None, observer_space, action_space, {"agent_id": 0, }),
            "pol1": (None, observer_space, action_space, {"agent_id": 1, }),
        },
        # Function mapping agent ids to policy ids
        "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1",
        # See rllib/evaluation/observation_function.py for more info
        "observation_fn": central_critic_observer,
    }

    #### Tuner Callbacks #######################################
    tuner_callbacks = [
        WandbLoggerCallback(project=f"{PROJECT_NAME}-trials",
                            save_checkpoints=True,
                            # log_config=True
                            )
    ]

    #### Ray Tune stopping conditions ##########################
    stop = {
        "timesteps_total": 5,  # 100000 ~= 10'
        # "episode_reward_mean": 0,
        # "training_iteration": 0,
    }

    #### Train #################################################
    results = tune.run(
        "PPO",
        stop=stop,
        config=config,
        verbose=True,
        checkpoint_at_end=True,
        local_dir=filename,
        callbacks=tuner_callbacks
    )
    # check_learning_achieved(results, 1.0)

    #### Save agent ############################################
    run = wandb.init(project=PROJECT_NAME, name=f"eval-{exp}", config=config)
    best_model_artifact = wandb.Artifact(exp, type="model")

    # Grab all the logged videos during training and log them during eval
    # TODO: On episode end of training log the video as opposed to after the fact
    training_video_table = video_recordings_to_wandb_table(
        videos_folder, fr=15)
    run.log({
        "training_videos": training_video_table
    })
    best_model_artifact.add(training_video_table, "training_videos")

    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                   mode='max'
                                                                                   ),
                                                      metric='episode_reward_mean'
                                                      )
    best_checkpoint_path = checkpoints[0][0]
    best_model_artifact.add_dir(best_checkpoint_path, name="model")

    #### Run best model in test environment ####################
    agent = ppo.PPOTrainer(config=config)
    agent.restore(best_checkpoint_path)

    #### Extract and print policies ############################
    policy0 = agent.get_policy("pol0")
    print("action model 0", policy0.model.action_model)
    print("value model 0", policy0.model.value_model)
    policy1 = agent.get_policy("pol1")
    print("action model 1", policy1.model.action_model)
    print("value model 1", policy1.model.value_model)

    #### Create test environment ###############################
    if ARGS.env == 'flock':
        test_env = FlockAviary(num_drones=ARGS.num_drones,
                               aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                               obs=ARGS.obs,
                               act=ARGS.act,
                               record=ARGS.record,
                               output_folder=eval_videos
                               )
    elif ARGS.env == 'leaderfollower':
        test_env = LeaderFollowerAviary(num_drones=ARGS.num_drones,
                                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                        obs=ARGS.obs,
                                        act=ARGS.act,
                                        record=ARGS.record,
                                        output_folder=eval_videos
                                        )
    elif ARGS.env == 'meetup':
        test_env = MeetupAviary(num_drones=ARGS.num_drones,
                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                obs=ARGS.obs,
                                act=ARGS.act,
                                record=ARGS.record,
                                output_folder=eval_videos
                                )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    #### Show, record a video, and log the model's performance #
    obs = test_env.reset()
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones,
                    colab=True,
                    output_folder=eval_logs
                    )
    if ARGS.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        action = {i: np.array([0]) for i in range(ARGS.num_drones)}
    elif ARGS.act in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        action = {i: np.array([0, 0, 0, 0]) for i in range(ARGS.num_drones)}
    elif ARGS.act == ActionType.PID:
        action = {i: np.array([0, 0, 0]) for i in range(ARGS.num_drones)}
    else:
        print("[ERROR] unknown ActionType")
        exit()
    start = time.time()
    for i in range(6*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)):  # Up to 6''
        #### Deploy the policies ###################################
        temp = {}
        # Counterintuitive order, check params.json
        temp[0] = policy0.compute_single_action(
            np.hstack([action[1], obs[1], obs[0]]))
        temp[1] = policy1.compute_single_action(
            np.hstack([action[0], obs[0], obs[1]]))
        action = {0: temp[0][0], 1: temp[1][0]}
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        print("~~~~~~~~~~~~~~~~~")
        print(len(obs))
        print(len(action))
        if ARGS.obs == ObservationType.KIN:
            for j in range(ARGS.num_drones):
                logger.log(drone=j,
                           timestamp=i/test_env.SIM_FREQ,
                           state=np.hstack([obs[j][0:3], np.zeros(
                               4), obs[j][3:15], np.resize(action[j], (4))]),
                           control=np.zeros(12)
                           )
        # sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
    test_env.close()

    #Use the logger to output results of evaluation and log them to W&B#
    csv_dir = logger.save_as_csv("ma")
    fig = logger.plot()

    best_model_artifact.add_dir(csv_dir, name="logs")
    output_figure = wandb.Plotly(fig)
    eval_video_table = video_recordings_to_wandb_table(eval_videos, fr=15)
    run.log({
        "eval_videos": eval_video_table,
        "eval_logs": output_figure
    })
    best_model_artifact.add(eval_video_table, "eval_videos")

    run.log_artifact(best_model_artifact)

    #### Shut down Ray #########################################
    ray.shutdown()
    run.finish()
