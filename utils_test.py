from pettingzoo.atari import pong_v3, space_invaders_v2
from utils.agent_indication_atari_wrapper import AgentIndicatorAtariEnv
env = space_invaders_v2.parallel_env(obs_type='ram')
env = AgentIndicatorAtariEnv(env)
observations, _ = env.reset()
print(observations['first_0'])
print(observations['second_0'])

for episode in range(200):
    action1 = int(input("first agent:"))
    action2 = int(input("second_agent:"))
    actions = {'first_0': action1, 'second_0': action2}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations['first_0'])
    print(observations['second_0'])