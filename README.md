# Napoleon
Lichess bot equipped with a Monte-Carlo Tree Search engine. The engine has been given several positions from the Lichess Elite Database (https://database.nikonoel.fr/). It explored each position using the standard steps of the MCTS (Selection, Expansion, Simulation and Backprogation) guided by an UCT score. The generated data was then used to train the neural network to predict the value of the position and the simulation policy. This cycle has been repeated several times using a VM on Google Cloud Platform. The bridge between the engine and the Lichess API has been cloned from https://github.com/lichess-bot-devs/lichess-bot. 


## Follow the steps of the original repository to run the bot.
1. Download this repository.
2. [Install and setup virtualenv](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Install) **but do not delete or modify the config file at this step.**
3. [Create a lichess OAuth token](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token)
4. [Upgrade to a BOT account](https://github.com/lichess-bot-devs/lichess-bot/wiki/Upgrade-to-a-BOT-account)
5. [Run lichess-bot](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Run-lichess%E2%80%90bot)

## Customize the depth
You can customize the number of times the MCTS algorithm will run by modifying the `depth` parameter in `homemade.py`. The higher the depth the better the bot will play. However, the bot will play slower.

## Customize the way the bot selects the move it will play.
The bot will use both the neural network's evaluations of the moves and the result of the MCTS simulations to decide what move to play. The `exploration` parameter in `homemade.py`, is a number between 0 and 1 that represents the weight of the simulations' result in the bot's decision.
