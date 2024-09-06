# At the end of the simulation phase, propagate the results of the simulation to all the concerned nodes.

from .node_main import node
import chess

def backpropagation(self, pos: chess.Board, exploration):

    if not pos.is_checkmate():

        self.update(exploration, 0)
        x = self
        while True:
            x = x.father
            x.update(exploration, 0)
            if x.father == None:
                break

    if pos.is_checkmate():

        if pos.turn:
            reward = -1
            self.update(exploration, reward)
            x = self
            while True:
                x = x.father
                x.update(exploration, reward)
                if x.father == None:
                    break
        
        else:
            reward = 1
            self.update(exploration, reward)
            x = self
            while True:
                x = x.father
                x.update(exploration, reward)
                if x.father == None:
                    break

node.backpropagation = backpropagation