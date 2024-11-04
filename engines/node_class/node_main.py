# Each chess board position is stored in a node

import random
import copy as copy
import numpy as np
import keras
import chess
import tensorflow as tf

class node:

    def __init__(self, model: keras.src.models.functional.Functional, position: chess.Board, move, father: "node", player: int):

        self.model = model # Neural Network
        self.position = position # Position with the library python-chess
        self.moves_to_explore = list(self.position.legal_moves) # List of all legal moves
        random.shuffle(self.moves_to_explore)
        self.father = father # Father node
        self.player = player # 1 if White -1 if Black
        self.move = move # Move from the father node
        self.index = None 
        if self.father != None:
            self.index = self.father.moves_to_explore.index(self.move) 
        if self.father == None: # If root node
            self.visits = 0
            self.array = self.state_representation()
        
        self.sons = [] # List of sons
        l = len(self.moves_to_explore)
        self.sons_proba = np.zeros(l) # Probability of winning
        self.sons_value = np.zeros(l) # Number of wins of sons
        self.sons_visits = np.zeros(l)
        self.sons_ucb = None
        self.sons_v = None  # List of the neural network evalutation of the node's sons
        self.sons_arrays = None # List containing the state representation of the node's sons

    def state_representation(self):
        pass

    def comp_usb(self, exploration): # Computes the UCB score of the node's all sons
        self.sons_ucb = self.player * (1-exploration) * self.sons_v + self.sons_proba * exploration + np.log(1 + self.father.sons_visits[self.index]) / (1 + self.sons_visits)

    def update(self, exploration: float, result: int): # Update the properties of the node and its sons
        pass

    def best_son(self): # Return the son with the highest UCB value
          return self.sons[np.argmax(self.sons_ucb)]
    
    def is_leaf(self): # Has the node ever been visited ?

          if not self.sons : # If there are no sons
              return True
          else :
              return False
          
    def selection(self): # Goes through the tree, from the root to a leaf node, by choosing the nodes with the best UCB score
          
          if self.is_leaf() == True:
              return self
          
          leaf = False
          checked_node = self

          while leaf == False :
               checked_node = checked_node.best_son()
               leaf = checked_node.is_leaf() 
                   
          return checked_node

    def blooming(self):
        pass

    def expansion(self): # Adds sons to a node without sons
        
        if self.position.outcome() == None:
            self.blooming()
            new_son = self.best_son()
            return new_son
        else:
            return self

    def simulation(self): # Plays randomly until reaching the end of the game
        
        if self.position.outcome() != None :
            return self.position
            
        board = copy.deepcopy(self.position)
        
        while board.outcome() == None:
            moves_to_explore = list(board.legal_moves)
            board.push(random.choices(moves_to_explore)[0])

        return board
            
    def backpropagation(self, pos: chess.Board, exploration):
        pass

    def choose_son(self, exploration):
        self.sons_ucb = self.player * (1-exploration) * self.sons_v + self.sons_proba * exploration
        return self.sons[np.argmax(self.sons_ucb)]
