import random
import copy as copy
import numpy as np
import keras
import chess
import tensorflow as tf

@keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):

    # Value loss (MSE)
    value_loss = tf.reduce_mean(tf.abs(y_true[0] - y_pred[0]))
    
    # Policy loss (categorical cross-entropy)
    policy_loss = tf.reduce_mean(tf.abs(y_true[1] - y_pred[1]))

    # Total loss
    total_loss = value_loss + policy_loss
    
    return total_loss

class node:

    def __init__(self, model: keras.src.models.functional.Functional, position: chess.Board, move, father: "node", player: int):

        self.model = model
        self.position = position
        self.moves_to_explore = list(self.position.legal_moves)
        random.shuffle(self.moves_to_explore)
        self.father = father
        self.player = player
        self.move = move
        self.index = None
        if self.father != None:
            self.index = self.father.moves_to_explore.index(self.move)
        if self.father == None:
            self.visits = 0
            self.array = self.array_builder()[0]
        
        self.sons = []
        l = len(self.moves_to_explore)
        self.sons_proba = np.zeros(l)
        self.sons_value = np.zeros(l)
        self.sons_normalized_value = np.zeros(l)
        self.sons_visits = np.zeros(l)
        self.sons_ucb = None
        self.sons_p = None
        self.sons_v = None
        self.sons_arrays = None
        self.sons_counts = None

    def comp_usb(self, exploration):
        self.sons_ucb = self.player * ((1-exploration) * self.sons_v + self.sons_normalized_value * exploration) +  (1 + self.sons_p)  * np.log(self.father.sons_visits[self.index]) / (1 + self.sons_visits)

    def update(self, exploration: float, result: int):

        if self.father == None:
            self.visits += 1
            if self.sons:
                self.sons_proba = self.sons_visits / self.visits
                self.sons_ucb = self.player * ((1-exploration) * self.sons_v + self.sons_normalized_value * exploration) + (1 + self.sons_p) * np.log(self.visits) / (1 + self.sons_visits) 

        else:

            self.father.sons_visits[self.index] +=  1
            if self.sons:
                self.sons_proba = self.sons_visits / self.father.sons_visits[self.index]

            if result == 1 :
                self.father.sons_value[self.index] +=  1
            elif result == -1 : 
                self.father.sons_value[self.index] +=  -1

            self.father.sons_normalized_value[self.index] = self.father.sons_value[self.index] / self.father.sons_visits[self.index]
        
            if self.sons:
                self.comp_usb(exploration)

    def best_son(self):
          return self.sons[np.argmax(self.sons_ucb)]
    
    def is_leaf(self):

          if not self.sons :
              return True
          else :
              return False
          
    def selection(self):
          
          if self.is_leaf() == True:
              return self
          
          leaf = False
          checked_node = self

          while leaf == False :
               checked_node = checked_node.best_son()
               leaf = checked_node.is_leaf() 
                   
          return checked_node

    def array_builder(self):

        array = np.zeros((8, 8)) # 8 x 8 array representation of the Board.
        dict = {
                    'P': 1, 'N': 3, 'B': 4, 'R': 5, 'Q': 9, 'K': 10,
                    'p': -1, 'n': -3, 'b': -4, 'r': -5, 'q': -9, 'k': -10}
        tcid = {
                    1: 12, 3: 13, 4: 14, 5: 15, 9: 16, 10: 17,
                    -1: 18, -3: 19, -4: 20, -5: 21, -9: 22, -10: 23}     
        
        values = np.array([-10, -9, -5, -4, -3, -1, 1, 3, 4, 5, 9, 10])

        if self.player == 1:
            player = True
        else:
            player = False 

            # Split the FEN by '/' 
        rows = self.position.fen().split(' ')[0].split('/')

        for i, row in enumerate(rows): #Convert the FEN to a 8 x 8 numpy array
            column = 0
            for letter in row:
                if letter.isdigit():
                    column += int(letter)
                else:
                    array[i, column] = dict[(letter)]
                    column += 1
        
        # Initialize a 3D array
        channels = np.zeros((8, 8, 52))
        counts = np.zeros(55)
        
        # Create a mapping from unique value to index
        value_to_index = {value: idx for idx, value in enumerate(values)}
        
        # Populate the 3D array
        for i in range(8):
            for j in range(8):

                value = array[i][j]
                sq = chess.square(j,7-i)

                if value != 0:

                    index = value_to_index[value]
                    points = np.sign(value) * self.player
                    channels[i, j, index] = points

                    attacks = self.position.attacks(sq)
                    attackers_1 = self.position.attackers(player, sq)
                    attackers_2 = self.position.attackers(not player, sq)
                    
                    for square in attacks:
                        row, col = divmod(square, 8)
                        channels[7 - row, col, tcid[value]] += points

                    for square in attackers_1:
                        row, col = divmod(square, 8)
                        channels[7 - row, col, 12 + tcid[value]] += self.player

                    for square in attackers_2:
                        row, col = divmod(square, 8)
                        channels[7 - row, col, 24 + tcid[value]] -= self.player
                    
                    for key in tcid.keys():
                        channels[:, :, 36 + key] = channels[:, :, 12 + key] + channels[:, :, 24 + key]

                channels[i, j, 49] = len(self.position.attackers(player, sq))
                channels[i, j, 50] = len(self.position.attackers(not player, sq))
                channels[i, j, 51] = len(self.position.attacks(sq))

        for i in range(52):
            counts[i] = np.count_nonzero(channels[:,:,i])
        
        counts[52] = len(self.moves_to_explore)
        
        if self.position.is_check():
            counts[53] = 1
            counts[54] = len(self.position.checkers())

        return channels, counts

    def blooming(self):

        for move in self.moves_to_explore:
            new_position = copy.deepcopy(self.position)
            new_position.push(move)
            self.sons.append(node(self.model, new_position, move, self, - self.player))
        
        sons_arrays = []
        sons_counts = []
        for son in self.sons:
            build = son.array_builder()
            sons_arrays.append(build[0])
            sons_counts.append(build[1])
        self.sons_arrays = np.array(sons_arrays)
        self.sons_counts = np.array(sons_counts)

        l = len(self.sons)
        p = np.full((l,),-self.player)
        if self.father == None:
            a = self.array
        else:
            a = self.father.sons_arrays[self.index]
        f = np.full((l, 8, 8, 52), a)

        self.sons_v, self.sons_p = self.model.predict([self.sons_arrays, p, f, self.sons_counts], verbose = 0)
        self.sons_p = self.sons_p.reshape(l,)
        self.sons_v = self.sons_v.reshape(l,)

        if self.father == None:
            self.sons_ucb =  self.player * self.sons_v + self.sons_p 
        else:
            self.sons_ucb =  self.player * self.sons_v + self.sons_p 

    def expansion(self):
        
        if self.position.outcome() == None:
            self.blooming()
            new_son = self.best_son()
            return new_son
        else:
            return self

    def simulation(self):
        
        if self.position.outcome() != None :
            return self.position, 0
        N = 0
        board = copy.deepcopy(self.position)
        while board.outcome() == None:
            N = N +1
            moves_to_explore = list(board.legal_moves)
            board.push(random.choices(moves_to_explore)[0])

        return board, N #N is the number of generation after which the game end.
            
    def backpropagation(self, pos: chess.Board, exploration, N: int):

        if not pos.is_checkmate():
 
            self.update(exploration, 0)
            x = self
            while True:
                x = x.father
                x.update(exploration, 0)
                if x.father == None:
                    break

        if pos.is_checkmate():

            if (N%2) != 0:
                reward = 1
                self.update(exploration, reward)
                x = self
                while True:
                    x = x.father
                    reward = -reward
                    x.update(exploration, reward)
                    if x.father == None:
                        break
         
            if (N%2) == 0:
                reward = -1
                self.update(exploration, reward)
                x = self
                while True:
                    x = x.father
                    reward = -reward
                    x.update(exploration, reward)
                    if x.father == None:
                        break

    def choose_son(self, exploration):
        self.sons_ucb = self.player * ((1-exploration) * self.sons_v + self.sons_value * exploration)
        return self.sons[np.argmax(self.sons_ucb)]

class MCTS:

    def __init__(self, model: keras.src.models.functional.Functional, starting_position: chess.Board, depth: int, exploration: float, memory):
        
        self.model = model

        if starting_position.turn:
            player = 1
        else:
            player = -1
        if memory == None:    
            self.root = node(self.model, starting_position, None, None, player)
        else:
            self.root = memory
            memory.father = None
        self.depth = depth #How many searches
        self.exploration = exploration #How much should it explores instead of prioritizing winning ratios.

    def investigate(self):

        for i in range(self.depth):
            print("Depth = ", i )
            choosen_node = self.root.selection()
            #print("Selection")
            new_son = choosen_node.expansion()
            #print("Expansion")
            position, generation = new_son.simulation()
            #print("Simulation")
            new_son.backpropagation(position, self.exploration, generation)
            #print("Backpropagation")


    def play_against_opponent(self):
        self.investigate()
        return self.root.choose_son(self.exploration)

def get_move(model, board, depth, exploration, memory):

    game = MCTS(model, board, depth, exploration, memory)
    memory = game.play_against_opponent()
    return memory.move

