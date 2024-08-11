# Napoleon
Lichess bot equipped with a Monte-Carlo Tree Search engine and a neural network that gives an evaluation of the position. The bridge between the engine and the Lichess API has been cloned from https://github.com/lichess-bot-devs/lichess-bot. 

## Neural Network

### Input
The neural network as input a (8, 8, 46) numpy array representating the position to evaluate. 

* 1st-12th channels are boards only containing the positions of a piece type (white pawns, black pawns, white queen, black queen, etc.). A piece belonging to the player to white is marked with +1 while black pieces are marked by -1.
* The 13th to 24th channels are boards that contain the squares attacked by a specific piece type. Each time a square is attacked by a piece, its array element is incremented with 1 or -1 depending of the piece's color.
* The 25th to 30th are element wise sums of the 13th to 24th channels. We sum channels of pieces of opposing color but belonging to the same species.
* The 31th channel is a sum of the 25th to 30th channels.
* The 32th channel is a board whose elements are the number of piece belonging to white attacking each square.
* The 33th channel is a board whose elements are the number of piece belonging to black attacking each square multiplied by -1.
* The 34th channel is a board whose elements are the number of squares attacked from each square. 

The board is rotated so that the neural network sees the board from the perspective of the player who has to make a move.

### Architecture

#### Squeeze and Excitation block

* Flatten instead of the usual pooling to not loose any information. When analyzing image, we have a large quantity of information and pooling allows to not have to unnecessarily use all the information. Chessboard are much smaller than images and using all the information doesn't have a high cost.
* Two denses layers (16 units and 34 units) with relu activation function, each layer followed by batch normalization.
* Multiply the output with the input array.

#### Residual Block type 1

* Batch normalization.
* Squeeze and Excitation block (without batch normalization and with tanh activation instead of relu).
* Add the result to the residual block's input.
* Apply tanh.
* Batch normalization.

#### Residual Block type 2

* Convolutional layer (34 filters, 3x3) with relu activation function.
* Batch normalization.
* Squeeze and Excitation block.
* Add the result with to the the residual block's input.
* Apply relu.
* Batch normalization.

#### Value head

* Flatten
* Denses layers with 16 units with relu activation function followed by batch normalization.
* Dense layer with one unit and tanh activation function.

The architecture is the following :

* 1 x Residual Block type 1
* 2 x Residual Block type 2
* Value head

#### Output

The network output a value between -1 and 1. A positive value means that white is winning while a negative value means that black is winning.

#### Training

The engine has been given 280 000 positions evaluated by stockfish (20% have been used to evaluate the network). The data has been taken from https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations. Non numerical evaluations and evaluations with absolute value higher than 10 have been excluded. The stockfish evaluations have then normalized to one and the neural network was tasked with deducting these values using the positions. 

### Monte-Carlo Tree Search 

The Monte-Carlo Tree Search opperate using the followin UCB score to navigate nodes :

$$ UCB = player * [(1-exploration) \times prediction + exploration \times value] + \frac{\log(1+N)}{1+n}  $$

* Player is 1 if white to play and -1 if black to play.
* Prediction is the neural network evaluation of the node's position.
* Value = Is the evaluation from the MCTS' simulations.
* Exploration is a parameter between 0 and 1.
* N is the number of times the father node has been visitied
* n is the number of times the node has been visited. 

For more on MCTS : https://www.chessprogramming.org/Monte-Carlo_Tree_Search

# Follow the steps of the original repository to run the bot.
1. Download this repository.
2. [Install and setup virtualenv](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Install) **but do not delete or modify the config file at this step.**
3. [Create a lichess OAuth token](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token)
4. [Upgrade to a BOT account](https://github.com/lichess-bot-devs/lichess-bot/wiki/Upgrade-to-a-BOT-account)
5. [Run lichess-bot](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Run-lichess%E2%80%90bot)

# Customize the depth
You can customize the number of times the MCTS algorithm will run by modifying the `depth` parameter in `homemade.py`. The higher the depth the better the bot will play. However, the bot will play slower.

# Customize the way the bot selects the move it will play.
The bot will use both the neural network's evaluations of the moves and the result of the MCTS simulations to decide what move to play. The `exploration` parameter in `homemade.py`, is a number between 0 and 1 that represents the weight of the simulations' result in the bot's decision.
