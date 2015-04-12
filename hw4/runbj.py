from blackjack import BlackjackEnv, BlackjackTask
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q
from pybrain.rl.experiments import Experiment, EpisodicExperiment
from pybrain.rl.explorers import EpsilonGreedyExplorer
import numpy as np

# define action-value table
# number of states is:
#
#    current value: 1-21
#
# number of actions:
#
#    Stand=0, Hit=1
controller = ActionValueTable(23, 2)
controller.initialize(0.)

# define Q-learning agent
learner = Q(0.5, 0.0)
learner._setExplorer(EpsilonGreedyExplorer(0.2))
agent = LearningAgent(controller, learner)

# define the environment
env = BlackjackEnv()

# define the task
task = BlackjackTask(env)

# finally, define experiment
experiment = EpisodicExperiment(task, agent)

# ready to go, start the process
for i in range(1):
    print "Iteration: {}".format(i)
    print experiment.doEpisodes(500)
    agent.learn(500)
    agent.reset()
    print ""

print np.round(controller.params.reshape(23, 2), 3)
