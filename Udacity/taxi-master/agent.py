import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        #self.policy = defaultdict(lambda: np.ones(self.nA))
        self.alpha = 0.2
        self.gamma = 1.0
        self.epsilon = 0.005
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.policy = self.update_Policy(state)
        
        return np.random.choice(np.arange(self.nA), p=self.policy)

    def step(self, state, action, reward, next_state, done, i_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.epsilon = 1 / i_episode
        
        
        self.Q[state][action] =  self.Q[state][action] + self.alpha * ((self.gamma * self.Q[next_state][action]) - self.Q[state][action])
        
     
    def update_Policy(self, state):
        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon  / self.nA) 
        return policy
                                                                          
                                                                                                         