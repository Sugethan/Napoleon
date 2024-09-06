# Methode building the representation of the chessboard for the neural network

from .node_main import node
import chess
import numpy as np

def state_representation(self):

    array = np.zeros((8, 8)) # 8 x 8 array representation of the Board.
    
    dict = {
                'P': 1, 'N': 3, 'B': 4, 'R': 5, 'Q': 9, 'K': 10,
                'p': -1, 'n': -3, 'b': -4, 'r': -5, 'q': -9, 'k': -10}
    tcid = {
                1: 12, 3: 13, 4: 14, 5: 15, 9: 16, 10: 17,
                -1: 18, -3: 19, -4: 20, -5: 21, -9: 22, -10: 23}      
    
    values = np.array([-10, -9, -5, -4, -3, -1, 1, 3, 4, 5, 9, 10])

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
                        
    if not self.position.turn:
        array = np.rot90(array, 2)
    
    # Initialize a 3D array
    channels = np.zeros((8, 8, 34))
    
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
                
                for square in attacks:
                    row, col = divmod(square, 8)
                    channels[7 - row, col, tcid[value]] += points

            attackers_1 = self.position.attackers(True, sq)
            attackers_2 = self.position.attackers(False, sq)

            for k in range(6):
                channels[:, :, 24 + k] = channels[:, :, 12 + k] + channels[:, :, 18 + k]
                channels[:, :, 30] = channels[:, :, 30] + channels[:, :, 24 + k]
            
            channels[i, j, 31] = len(attackers_1)
            channels[i, j, 32] = - len(attackers_2)
            channels[i, j, 33] = len(self.position.attacks(sq))

    return channels

node.state_representation = state_representation