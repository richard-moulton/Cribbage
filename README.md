# Cribbage
This project contains the code for Cribbage as a reinforcement learning problem.

## To get right into it
The agents are: DeepPeg.py, LinearB.py, Myrmidon.py, Monty.py, Monty2.py, NonLinearB.py, and PlayerRandom.py

Each agent can be run directly as a python script
```>> python3 <agentname>.py```
This will run the agent in the Arena against Myrmidon and produces learning curve performance graphs.

Supporting files are: TrainHand.py, TrainPegging.py, and TrainingScript.py

These files are configured to be run directly as scripts; TrainHand.py and TrainPegging.py are both interactive scripts.

## More detailed file rundown
There are four kinds of files in this folder.

### CRIBBAGE FILES
1. Cribbage.py: Main file for playing a game of cribbage. Plays through hands one at a time, scoring for nobs, pegging and hands as it goes. Winner is declared as soon as one player reaches 121 points.
2. Deck.py: Classes representing Suits, Ranks, Cards and Decks.
3. Scoring.py: Scores cards according to the rules of cribbage.

### PLAYER FILES
4. Player.py: An abstract class defining what methods a player class must have in order to play well with Cribbage.py.
5. PlayerRandom.py: A simple instantiation of a Player. Makes decisions randomly.
6. Myrmidon.py: A Player that makes use of one-step rollouts and heuristics.
7. LinearB.py: A Player that represents hands using a linear combination of features. These features are then used for episodic semi-gradient one-step Sarsa during the throwing cards phase and for true online Sarsa during the pegging phase.
8. NonLinearB: A Player that represents hands using a non-linear combination of features. These features are then used for episodic semi-gradient one-step Sarsa during the throwing cards phase and for true online Sarsa during the pegging phase.
9. DeepPeg: A Player that uses two multilayer perceptron regressors to encode Q values: one for pegging and one for throwing cards.
10. Monty.py: A player that uses first visit Monte Carlo to learn the Q values for different states. A minor modification of QLearner.
11. Monty2.py: A second player that uses first visit Monte Carlo to learn the Q values for different states. A minor modification of QLearner.

### GENERAL FILES
12. Arena.py: Records performance data for a player over a number of hands. Can be used to produce training curve data or to measure final performance levels.
13. TrainPegging.py: Trains players on pegging phase of Cribbage
14. TrainHand.py: Trains players on a single hand of Cribbage
15. TrainingScript.py: Automates the processes of training agents against each other and of running round robin tournaments.
16. Utilities.py: Useful functions that are used throughout the project.

### MEMORY FILES
A number of learning agents store parameters in files. These are:
throwWeights.npy and pegWeights.npy: LinearB
NLBthrowWeights.npy and NLBpegWeights.npy: NonLinearB
Brain files in the directory 'BrainsInJars': QLearner, Monty, and Monty2

# Dependencies
These files have a number of dependencies on standard python libraries:

* abc
* enum
* math
* numpy
* sklearn
* os
* joblib
* warnings
* itertools
* matplotlib
