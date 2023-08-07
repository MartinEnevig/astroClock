from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
#from gymnasium.wrappers import FlattenObservation
from env import spaceEnv

envy = spaceEnv(full_mode=True)
envy.reset()

#Uncomment, if you need to check the env
check_env(envy)


#do x steps in the env with rtandom actions
# x = 15
# for i in range(x):
#     envy.step(action=envy.action_space.sample())






