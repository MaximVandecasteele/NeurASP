# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from stable_baselines3_master.stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy