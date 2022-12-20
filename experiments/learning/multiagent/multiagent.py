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
import torch
import ray
from ray import air, tune
from ray.tune.stopper import (CombinedStopper,
    MaximumIterationStopper, TrialPlateauStopper)
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from gym.spaces import Box, Dict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

from ray.air.integrations.wandb import WandbLoggerCallback  # 🪄🐝
import wandb

from gym_pybullet_drones.utils.enums import DroneModel, Physics

from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger

from utils import video_recordings_to_wandb_table, configure_and_create_folders
from model import CustomTorchCentralizedCriticModel, configure_constants
from env_manager import create_env

import multiprocessing

SEED = 1
############################################################
if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=2,                 type=int,
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
    exp, filename, videos_folder, eval_folder, eval_videos, eval_logs = configure_and_create_folders(
        output_folder, ARGS)
    print("Configured folders")

    #### Constants, and errors #################################
    OWN_OBS_VEC_SIZE, ACTION_VEC_SIZE = configure_constants(ARGS)
    
    ####
    #TODO: Move to model.py
    #BUG:  OWN_OBS_VEC_SIZE, ACTION_VEC_SIZE global variables not properly being accessed after being modified
    ######DEFINE ADDITIONAL MODEL STATE INFO########################
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

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the custom centralized critic model ##########
    ModelCatalog.register_custom_model(
        "cc_model", CustomTorchCentralizedCriticModel)
    print("Registered model")

    #### Register the environment ##############################
    temp_env_name = "this-aviary-v0"
    register_env(temp_env_name, lambda _: create_env(videos_folder, ARGS))
    temp_env = create_env(videos_folder, ARGS)
    print("Registered environment")
    #### Unused env to extract the act and obs spaces ##########
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
    
    
    # 🔦+🔎 Get all your useful hardware information
    use_gpu = torch.cuda.is_available()
    num_cpus = multiprocessing.cpu_count()

    # 🧮 Do some quick math to efficiently spread the workload to each GPU
    if use_gpu:
        num_gpus = torch.cuda.device_count()
        num_rollout_workers = 4 if num_gpus>=4 else num_gpus
        num_gpus_per_worker = num_gpus // num_rollout_workers
        num_cpus_per_worker = 4 * num_gpus_per_worker #Assumes 4 cpus per gpu
    else:
        num_gpus = 0
        num_gpus_per_worker = 0
        num_cpus_per_worker = 4
        num_rollout_workers = num_cpus // num_cpus_per_worker
    
    #TODO: Make arguments
    algorithm = "PPO"
    train_batch_size = 128
    num_gpus = 2
    #TODO: Gets stuck on PENDING if resource starved
    num_rollout_workers = 1
    num_gpus_per_worker = 1
    num_cpus_per_worker = (num_cpus//num_rollout_workers) - 1
    
    
    print(f"""
    {num_cpus}
    {num_gpus}
    {num_rollout_workers}
    {num_gpus_per_worker}
    {num_cpus_per_worker}
    """
    )

    
    config = (
        AlgorithmConfig(algo_class=algorithm)
        .environment(temp_env_name)
        .framework("torch")
        .rollouts(
            num_rollout_workers=num_rollout_workers,
            # num_envs_per_worker=4,
            rollout_fragment_length="auto",
            batch_mode="complete_episodes"
        )
        .training(
            train_batch_size=train_batch_size,
            # gamma=0.9,
            model={
                "custom_model": "cc_model",
                "custom_model_config": {
                    "OWN_OBS_VEC_SIZE" : OWN_OBS_VEC_SIZE,
                    "ACTION_VEC_SIZE" : ACTION_VEC_SIZE,
                    "SEED": SEED
                }
            }
        )
        .callbacks(
            callbacks_class=FillInActions
        )
        .multi_agent(
            policies={
                "pol0": (None, observer_space, action_space, {"agent_id": 0, }),
                "pol1": (None, observer_space, action_space, {"agent_id": 1, }),
            },
            policy_mapping_fn=(
                lambda agent_id, episode, worker, **kw: (
                    "pol0" if agent_id == 0 else "pol1"
                )
            ),
            observation_fn=central_critic_observer,
            )
        .resources(num_gpus=num_gpus,
                   # num_gpus_per_worker=num_gpus_per_worker,
                  num_cpus_per_worker=num_cpus_per_worker
                  )
    )
    print(f"Created {algorithm} config")
    #### Tuner Callbacks #######################################
    tuner_callbacks = [
        WandbLoggerCallback(project=f"{PROJECT_NAME}-trials",
                            save_checkpoints=True,
                            log_config=True
                            )
    ]
    
    ### Tuner Checkpoint settings ####################################
    checkpoint_config = air.CheckpointConfig(
                # We'll keep the best five checkpoints at all times
                # checkpoints (by episode_reward_mean, reported by the trainable, descending)
                checkpoint_score_attribute="episode_reward_mean",
                num_to_keep=3,
            )
    
    #### Ray Tune stopping conditions ##########################
    stop = TrialPlateauStopper(metric="episode_reward_mean")
    
    #### Train #################################################
    #TODO: Add a tunable parameter on SEED
    model_tuner = tune.Tuner(
        algorithm, param_space=config, run_config=air.RunConfig(
            verbose=3, local_dir=filename,
            stop=stop,
            callbacks=tuner_callbacks,
            checkpoint_config=checkpoint_config)
    )
    print("Created model tuner")

    results_grid = model_tuner.fit()
    print("Trained model")
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
    print("Logged training videos")
    best_model_artifact.add(training_video_table, "training_videos")
    
    results = results_grid._experiment_analysis #TODO: Perform this logic directly with the tuner.ResultGrid as opposed to ExperimentAnalysis
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
    print(f"Restored best model from {best_checkpoint_path}")

    #### Extract and print policies ############################
    policy0 = agent.get_policy("pol0")
    print("action model 0", policy0.model.action_model)
    print("value model 0", policy0.model.value_model)
    policy1 = agent.get_policy("pol1")
    print("action model 1", policy1.model.action_model)
    print("value model 1", policy1.model.value_model)

    #### Create test environment ###############################
    test_env = create_env(eval_videos, ARGS)
    print("Created test environment")

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
    print("Logged evaluation details")
    best_model_artifact.add(eval_video_table, "eval_videos")

    run.log_artifact(best_model_artifact)
    print("Logged all details to model artifact")

    #### Shut down Ray #########################################
    ray.shutdown()
    run.finish()
    print("Finished!")