# Napoleon
Lichess bot equipped with a Monte-Carlo Tree Search engine and a neural network that gives an evaluation of the position and a policy-like value. The bridge between the engine and the Lichess API has been cloned from https://github.com/lichess-bot-devs/lichess-bot. 

## Neural Network

### Inputs
The neural network takes four inputs. The first two inputs are (8, 8, 15) numpy arrays representating the position to evaluate and the previous position. The first twelve channels are boards only containing one type of piece (white pawns, white queen, black bishops, etc.), the 13th and 14th channels counts the number of white and black pieces attacking each square and the 15th channel gives the number of squares attacked from each square. The third input is a (14,) numpy array whose first twelve entries are the number of each piece type on the board, the 13th entry tells the network if the position is a check and the 14th entry the number of checkers. The fourth input is the the color to play.

### Architecture
The boards are analyzed through convolution layers separated by squeeze and excitation blocks and batch normalization layers. The value head doesn't receive the previous position. All layers use a ReLu activation functions execpt the value and policy output layers who use respectively tanh and sigmoid.

### Training
The engine has been given several positions from the Lichess Elite Database (https://database.nikonoel.fr/). It explored each position using the standard steps of the MCTS (Selection, Expansion, Simulation and Backprogation) guided by an UCT score. The generated data was then used to train the neural network to predict the value of the position and the simulation policy. This cycle has been repeated several times using a VM on Google Cloud Platform. The data used to train the neural network has been self generated. However due to time and material limits, the neural network has not finished its training.

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
