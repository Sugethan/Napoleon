from .node_class.node_main import node
import copy as copy
import keras
import chess


class MCTS:

    def __init__(self, model: keras.src.models.functional.Functional, starting_position: chess.Board, depth: int, exploration: float, memory):
        
        self.model = model

        if memory == None:   
            if starting_position.turn:
                player = 1
            else:
                player = -1  
            self.root = node(self.model, starting_position, None, None, player)
        else:
            self.root = memory
            
        self.depth = depth #How many searches
        self.exploration = exploration #How much should it explores instead of prioritizing winning ratios.

    def investigate(self):

        for i in range(self.depth):
            print("Depth = ", i )
            choosen_node = self.root.selection()
            #print("Selection")
            new_son = choosen_node.expansion()
            #print("Expansion")
            position = new_son.simulation()
            #print("Simulation")
            new_son.backpropagation(position, self.exploration)
            #print("Backpropagation")


    def play_against_opponent(self):
        self.investigate()
        return self.root.choose_son(self.exploration)

def get_move(model, board, depth, exploration, memory):

    game = MCTS(model, board, depth, exploration, memory)
    memory = game.play_against_opponent()
    return memory.move

