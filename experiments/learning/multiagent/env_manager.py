import shared_constants
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary


def create_env(output_folder, ARGS):
    if ARGS.env == 'flock':
        temp_env = FlockAviary(num_drones=ARGS.num_drones,
                               aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                               obs=ARGS.obs,
                               act=ARGS.act,
                               record=ARGS.record,
                               output_folder=output_folder
                               )
    elif ARGS.env == 'leaderfollower':
        temp_env = LeaderFollowerAviary(num_drones=ARGS.num_drones,
                                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                        obs=ARGS.obs,
                                        act=ARGS.act,
                                        record=ARGS.record,
                                        output_folder=output_folder
                                        )
    elif ARGS.env == 'meetup':
        temp_env = MeetupAviary(num_drones=ARGS.num_drones,
                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                obs=ARGS.obs,
                                act=ARGS.act,
                                record=ARGS.record,
                                output_folder=output_folder
                                )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    return temp_env
