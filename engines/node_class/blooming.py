from .node_main import node
import copy
import numpy as np


def blooming(self):

    for move in self.moves_to_explore:
        new_position = copy.deepcopy(self.position)
        new_position.push(move)
        self.sons.append(node(self.model, new_position, move, self, - self.player))
    
    sons_arrays = []
    for son in self.sons:
        sons_arrays.append(son.state_representation())

    self.sons_arrays = np.array(sons_arrays)

    l = len(self.sons)

    self.sons_v = self.model.predict([self.sons_arrays], verbose = 0)
    self.sons_v = self.sons_v.reshape(l,)

    self.sons_ucb =  self.player * self.sons_v 

node.blooming = blooming