#Theophilus Braimoh (tob2zd)
#4/9/2023

import gym
from gym import spaces
import atari_py
import cv2,os,sys
from stable_baselines3 import PPO

# create an instance of the Asteroids-v5 environment
#------------------------------------------------------
env = gym.make('ALE/Asteroids-v5', render_mode='human') 
#------------------------------------------------------
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save("100kMLP")
model = PPO.load('100kMLP.zip')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model = PPO.save('200kMLP.zip')
#------------------------------------------------------
episodes = 10
max_timesteps = 1000
#------------------------------------------------------


for i in range(1,episodes):
    # reset the environment to start a new episode
    observation = env.reset()
    done = False
    timesteps = 0
    
    #open file to output log to
    # with open("PPO_logs.txt", "a") as f:
    while not done and timesteps < max_timesteps:
        # use the trained model to choose an action
        action, _ = model.predict(observation)
        # take a step in the environment
        observation, reward, done, info = env.step(action)

        log_string = f"Episode {i}, timestep {timesteps}: reward={reward}, done={done}, info={info}\n"            
        print(log_string)
        # f.write(log_string)
        timesteps += 1

    episode_string = f"Episode {i} finished after {timesteps} timesteps.\n"
    print(episode_string)
    # f.write(episode_string)

# close file and environment when done   
# f.close
env.close()
