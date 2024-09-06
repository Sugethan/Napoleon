# Update the UCB values, visitis counts and values of the node's son

from .node_main import node
import numpy as np

def update(self, exploration: float, result: int):

    if self.father == None: # If root node
        self.visits += 1
        if self.sons:
            self.sons_proba = self.sons_visits / self.visits
            self.sons_ucb = self.player * ((1-exploration) * self.sons_v + self.sons_normalized_value * exploration) + np.log(1 + self.visits) / (1 + self.sons_visits) 

    else:

        self.father.sons_visits[self.index] +=  1
        if self.sons:
            self.sons_proba = self.sons_visits / self.father.sons_visits[self.index]

        self.father.sons_value[self.index] +=  result


        self.father.sons_normalized_value[self.index] = self.father.sons_value[self.index] / self.father.sons_visits[self.index]
    
        if self.sons:
            self.comp_usb(exploration)

node.update = update