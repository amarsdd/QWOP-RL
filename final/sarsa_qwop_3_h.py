import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from time import time
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.callbacks import *


### HYPERPARAMETERS

# base network size: nb_observations x 64 x 32 x 16 x nb_actions

# scale ---- scaled network : nb_observations x (64*scale) x (32*scale) x (16*scale) x nb_actions
scale = 3
# nb_actions : number of possible actions : Q = Rotate hip in one direction
#                                           W = Rotate hip in opposite direction
#                                           O = Rotate both knee joints in opposite direction
#                                           P = Rotate both knee joints in opposite direction of O
#                                           *no key press = No action

# nb_obs : size of the observations space : 24 in this QWOP game including body angle, velocity, hip joint angle and speed, knee joints angle and speed etc

# warmup befor training
nb_steps_warmup = 1000

# learning rate
lrn_rate = 1e-3

# total training time in terms of number of steps (number of actions taken)
nb_steps = 1000000

# maximum number of steps in each episode
nb_max_episode_steps=2000

# setting up the OpenAI-GYM environment for the QWOP game
ENV_NAME = 'qwoph1-v0'
ENV_NAME = 'qwophardh1-v0'
#
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n

# agent network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(32*scale))
model.add(Activation('relu'))
model.add(Dense(16*scale))
model.add(Activation('relu'))
model.add(Dense(8*scale))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='softmax'))
print(model.summary())

# spcifications for the RL agent
policy = EpsGreedyQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=1000, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# compiling the model
sarsa.compile(Adam(lr=lrn_rate), metrics=['mae'])

# setting up callbacks for result collection and realtime visualization of the results through tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
tpl = TrainEpisodeLogger()

# finally perform the training----- visualize=False enables training without visualizing the game which speeds up the training process
sarsa.fit(env, nb_steps=nb_steps, visualize=False, verbose=2,  callbacks=[tensorboard, tpl], nb_max_episode_steps=nb_max_episode_steps)

# save the model weights
sarsa.save_weights('sarsa_%d_%s_weights.h5f' %(scale, ENV_NAME), overwrite=True)



# save the training results
metrics=[]
def dict_to_list(dc):
    re =[]
    for key in dc:
        re.append(dc[key])
    return re
tt = dict_to_list(tpl.rewards_mean)
mm = np.array(tt[:-1])
kk = dict_to_list(tpl.metrics_at_end)
jj = np.array(kk[:-1])
metrics = np.column_stack((mm, jj))

import pickle 
pickle.dump( metrics, open( 'sarsa_%d_%s_metrics.p' %(scale, ENV_NAME), "wb" ) )

# load model for testing
sarsa.load_weights('/home/am/Desktop/set_tests/final/sarsa_%d_%s_weights.h5f' %(scale, ENV_NAME))

# setting up monitoring tools to record the testing episodes
from gym import monitoring
from gym.wrappers import Monitor

def episode5(episode_id):
    if episode_id < 5:
        return True
    else:
        return False
#rec = StatsRecorder(env,"sarsa_1")
#rec.capture_frame()
    
temp = '/home/am/Desktop/set_tests/final/sarsa_%d_%s' %(scale, ENV_NAME)
env = Monitor(env, temp, force=True,video_callable=episode5)

# testing
sarsa.test(env, nb_episodes=5, visualize=False,  nb_max_episode_steps=2000)

env.close()
results = monitoring.load_results(temp)



