from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from gym.spaces import Box, Dict
import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType


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


def configure_constants(ARGS):
    global OWN_OBS_VEC_SIZE, ACTION_VEC_SIZE
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
