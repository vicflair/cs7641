from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task
from pybrain.rl.environments import EpisodicTask
from scipy import zeros, clip, asarray
from pybrain.rl.environments.task import Task
import numpy as np


class BlackjackEnv(Environment):

    """ A (terribly simplified) Blackjack game implementation of an environment. """

    # Current hand
    hand = None

    # Deck to draw from. Aces count as 11 always.
    deck = range(2, 12) + [10]*3

    # Done with hand?
    done = False

    def getSensors(self):
        return [self.hand]

    def performAction(self, action):
        if action == 1:
            self.hand += np.random.choice(self.deck)
            if self.hand > 21:
                self.hand = 22
                self.done = True
            print "Hand: {}".format(self.hand)
        elif action == 0:
            self.done = True

    def reset(self):
        """ Most environments will implement this optional method that allows for reinitialization.
        """
        self.hand = sum(np.random.choice(self.deck, 2))
        print 'Opening hand: {}'.format(self.hand)
        self.done = False


class BlackjackTask(EpisodicTask):

    """ A task is associating a purpose with an environment. It decides how to evaluate the  observations, potentially returning reinforcement rewards or fitness values. """

    def __init__(self, environment):
        """ All tasks are coupled to an environment. """
        self.env = environment
        #self.reset()
        # we will store the last reward given, remember that "r" in the Q
        # learning formula is the one from the last interaction, not the one
        # given for the current interaction!
        self.lastreward = 0

    def performAction(self, action):
        """ A filtered mapping towards performAction of the underlying environment. """
        print "Action: {}".format(action)
        self.env.performAction(action)

    def getObservation(self):
        """ A filtered mapping to getSample of the underlying environment. """
        sensors = self.env.getSensors()
        return sensors

    def getReward(self):
        """ Compute and return the current reward (i.e. corresponding to the last action performed) """
        if self.env.done == True:
            dealer_hand = self.dealer()
            print "Dealer's hand: {}".format(dealer_hand)
            if self.env.hand == 22:  # Check for player bust first
                reward = -1
            elif (dealer_hand == 22) or (self.env.hand > dealer_hand):
                reward = 1
            elif self.env.hand == dealer_hand:
                reward = 0
            else:
                reward = -1
            #self.env.reset()
        else:
            reward = 0

        # retrieve last reward, and save current given reward
        #cur_reward = self.lastreward
        #self.lastreward = reward

        return reward

    def getReward2(self):
        """ Compute and return the current reward (i.e. corresponding to the last action performed) """
        if self.env.done == True:
            if self.env.hand == 22:  # Check for player bust first
                reward = -1
            elif self.env.hand < 17:
                reward = self.dealer_hand_prob[0]
            else:
                reward = sum(self.dealer_hand_prob[:(self.env.hand-16)])
            #self.env.reset()
        else:
            reward = 0
        return reward

    def reset(self):
        self.env.reset()

    def dealer(self):
        return np.random.choice([22, 17, 18, 19, 20, 21], p=self.dealer_hand_prob)

    def isFinished(self):
        if self.env.done == True:
            return True
        else:
            return False

    @property
    def dealer_hand_prob(self):
        # http://www.dagnammit.com/odds/
        return [.2826, .1453, .1388, .1341, .1781, .1211]

