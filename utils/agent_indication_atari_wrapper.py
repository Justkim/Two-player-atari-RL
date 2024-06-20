from utils.annotations import atari_annotations
import numpy as np

from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class AgentIndicatorAtariEnv(BaseParallelWrapper):
    """Creates a new environment using the base environment that runs for `num_episodes` before truncating.

    This is useful for creating evaluation environments.
    When there are no more valid agents in the underlying environment, the environment is automatically reset.
    When this happens, the `observation` and `info` returned by `step()` are replaced with that of the reset environment.
    The result of this wrapper is that the environment is no longer Markovian around the environment reset.
    """

    def __init__(self, env: ParallelEnv):
        """__init__.

        Args:
            env (AECEnv): the base environment
            num_episodes (int): the number of episodes to run the underlying environment
        """
        super().__init__(env)
        assert isinstance(
            env, ParallelEnv
        ), "MultiEpisodeEnv is only compatible with ParallelEnv environments."

        self.env_name = self.env.metadata['name']
        self.list_of_pairs = []
        atari_dict = atari_annotations
        if self.env_name in atari_dict:
            game_dict = atari_dict[self.env_name]
            for key in game_dict.keys():
                pair = game_dict[key]
                self.list_of_pairs.append(pair)
        else:
            print("Env annotations not found!")
            exit()
        


    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """reset.

        Args:
            seed (int | None): seed for resetting the environment
            options (dict | None): options

        Returns:
            tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """
        obs, info = super().reset(seed=seed, options=options)
        obs['second_0'] = self.change_observation(obs['second_0'])
        return obs, info

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """Steps the environment.

        When there are no more valid agents in the underlying environment, the environment is automatically reset.
        When this happens, the `observation` and `info` returned by `step()` are replaced with that of the reset environment.
        The result of this wrapper is that the environment is no longer Markovian around the environment reset.

        Args:
            actions (dict[AgentID, ActionType]): dictionary mapping of `AgentID`s to actions

        Returns:
            tuple[
                dict[AgentID, ObsType],
                dict[AgentID, float],
                dict[AgentID, bool],
                dict[AgentID, bool],
                dict[AgentID, dict],
            ]:
        """
        obs, rew, term, trunc, info = super().step(actions)
        obs['second_0'] = self.change_observation(obs['second_0'])
        return obs, rew, term, trunc, info
    

    def change_observation(self, obs_input):
        obs = np.copy(obs_input)
        for pair in self.list_of_pairs:
            temp = obs_input[pair[1]]
            obs[pair[1]] = obs_input[pair[0]]
            obs[pair[0]] = temp
        return obs