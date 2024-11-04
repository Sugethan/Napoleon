# Update the UCB values, visitis counts and values of the node's son

from .node_main import node
import numpy as np

def update(self, exploration: float, result: int):

    if self.father == None: # If root node
        self.visits += 1
        if self.sons:
            self.sons_proba = self.sons_visits / self.visits
            self.sons_ucb = self.player * (1-exploration) * self.sons_v + self.sons_proba * exploration + np.log(1 + self.visits) / (1 + self.sons_visits) 

    else:

        self.father.sons_visits[self.index] +=  1
        if self.sons:
            self.sons_proba = self.sons_value / self.sons_visits
            self.comp_usb(exploration)

        self.father.sons_value[self.index] +=  result

node.update = update
