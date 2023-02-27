from gym.envs.registration import register

register(
    id="example-v0",
    entry_point="gym_example.envs:Example_v0",
)

register(
    id="lssa-v0",
    entry_point="gym_example.envs:LSSA_v0",
)

register(
    id="fail-v1",
    entry_point="gym_example.envs:Fail_v1",
)
