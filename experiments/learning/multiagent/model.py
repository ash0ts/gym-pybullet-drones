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
import random

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
    

    #TODO: Make this GPU compatible
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, OWN_OBS_VEC_SIZE, ACTION_VEC_SIZE, SEED):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_model = FullyConnectedNetwork(
            Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE, )),
            action_space,
            num_outputs,
            model_config,
            name + "_action"
        )#.to(self.device)
        self.value_model = FullyConnectedNetwork(
            obs_space,
            action_space,
            1,
            model_config,
            name + "_vf"
        )#.to(self.device)
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        value_out, _ = self.value_model(
            {"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])


def configure_constants(ARGS):
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
    return (OWN_OBS_VEC_SIZE, ACTION_VEC_SIZE)
