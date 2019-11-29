#!/usr/bin/env python3

################################################################################
#
# File : DeepPeg.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : A Player that uses two multilayer perceptron regressors to encode
#               Q values: one for pegging and one for throwing cards.
#
# Notes : The game state is encoded as follows:
#           - 5 elements for the agent's actions
#                - [first card in hand, second card, third, fourth, go]
#                - if no card in hand, value is 0. Hand is always sorted ascending
#                - on card's rank value (A=1, King=13)
#           - 8 elements for the cards previously played during pegging
#               - 0 if no card played in that slot
#           - 2 elements for the cards the agent threw into the crib at the start of pegging
#           - 1 element for the starter card
#           - 1 element for the current count (0/31 : 31/31)
#           - 1 element for the dealer (0 = I'm not the dealer, 1 = I'm the dealer)
#         
#         If this file is run, it instantiates an Arena and measures this agent's
#         performance against Myrmidon.
#
# Dependencies:
#    - Deck.py (in local project)
#    - Arena.py (in local project)
#    - Player.py (in local project)
#    - Myrmidon.py (in local project)       * - for __name__ = '__main__' only
#    - Utilities.py (in local project)
#    - numpy (standard python library)
#    - sklearn (standard python library)
#    - random (standard python library)
#    - os (standard python library)
#    - joblib (standard python library)
#    - itertools (standard python library)
#    - matplotlib (standard python library) * - for __name__ = '__main__' only
#
################################################################################

# Cribbage imports
from Deck import Card
from Arena import Arena

# Player imports
from Player import Player
from Myrmidon import Myrmidon

# Utility imports
from Utilities import *
import numpy as np
import sklearn as skl
from sklearn import neural_network as sknn
import random
import os
import joblib
from itertools import combinations
import matplotlib.pyplot as plt

class DeepPeg(Player):

    def __init__(self, number, saveBrains=True, verbose=False):
        super().__init__(number, verbose)
        self.alpha = 0.5
        # Note: epsilon unused since we use softmax decision making
        self.epsilon = 0.1
        self.gamma = 0.1
        self.name = 'DeepPeg'

        self.prevAction = 5
        self.currentAction = -1
        self.prevState = []
        self.prevScore = 0
        self.cribThrow = []
        self.throwingState = []
        self.saveFlag = saveBrains

        self.filedir = 'BrainsInJars'
        filename = 'QlearnV1.brain'
        self.fullfilepegging = os.path.join(os.getcwd(), self.filedir, filename)

        if os.path.exists(self.fullfilepegging):
            self.peggingBrain = joblib.load(self.fullfilepegging)
            print("Pegging Brain loaded")
        else:
            self.peggingBrain = sknn.MLPRegressor(hidden_layer_sizes=(50, 25, 15), activation='relu', solver='adam',
                                                  alpha=0.1,
                                                  batch_size=1, max_iter=200)
            print("New Pegging Brain created")

        filename = 'Qlearn_throwV1.brain'
        self.fullfilethrowing = os.path.join(os.getcwd(), self.filedir, filename)

        if os.path.exists(self.fullfilethrowing):
            self.throwingBrain = joblib.load(self.fullfilethrowing)
            print("Throwing Brain loaded")
        else:
            self.throwingBrain = sknn.MLPRegressor(hidden_layer_sizes=(25, 10), activation='relu', solver='adam',
                                                   alpha=0.1,
                                                   batch_size=1, max_iter=200)
            print("New Throwing Brain created")


    def backup(self):
        if not os.path.isdir(self.filedir):
            os.mkdir(self.filedir)
        joblib.dump(self.peggingBrain, self.fullfilepegging)
        joblib.dump(self.throwingBrain, self.fullfilethrowing)

    def throwCribCards(self, numCards, gameState):
        cribCards = []
        self.cribThrow = []
        cardIndices = list(range(0, len(self.hand)))
        maxValue = -np.inf
        if gameState['dealer'] == self.number - 1:
            dealerFlag = 1
        else:
            dealerFlag = 0

        for combination in combinations(cardIndices, len(self.hand) - numCards):
            handCards = []
            thrownCards = []
            for i in range(0, len(cardIndices)):
                if i in combination:
                    handCards.append(Card(self.hand[i].rank, self.hand[i].suit))
                else:
                    thrownCards.append(Card(self.hand[i].rank, self.hand[i].suit))

            q = self.throwValue(self.getThrowingFeatures(handCards, thrownCards, dealerFlag))

            if q > maxValue:
                maxValue = q
                cribCards = []
                cribCards.append(thrownCards.pop())
                cribCards.append(thrownCards.pop())

        for i in range(0, len(cribCards)):
            for j in range(0, len(self.hand)):
                if cribCards[i].isIdentical(self.hand[j]):
                    self.cribThrow.append(Card(self.hand[j].rank, self.hand[j].suit))
                    self.hand.pop(j)
                    break

        if (self.verbose):
            print("{} threw {} cards into the crib".format(self.getName(), numCards))

        super().createPlayHand()

        return cribCards

    def playCard(self, gameState):
        if len(self.playhand) == 4:
            self.throwingState = self.getCurrentState(gameState)
        # Have to store state here, since it depends on the playhand which changes below
        self.prevState = self.getCurrentState(gameState)

        actionIndex = self.chooseAction(gameState, True)

        if actionIndex == 4:
            # Go
            cardPlayed = None
        else:
            cardPlayed = self.playhand.pop(actionIndex)
            self.prevAction = actionIndex

        self.prevScore = self.getRelativeScore(gameState)
        return cardPlayed

    def chooseAction(self, gameState, softmaxFlag):
        # Make sure hand is sorted
        self.playhand.sort()
        # Figure out which actions are legal
        legal = [(gameState['count'] + card.value()) <= 31 for card in self.playhand]
        legalHand = [card for (card, isLegal) in zip(self.playhand, legal) if isLegal]
        # legal hand is a new hand of only the cards that can legally be played
        # If no legal cards, only option is Go.
        if not legalHand:
            actionIndex = [4]
        else:
            # Choosing a card from the legal cards
            state = self.getCurrentState(gameState)
            # Predict using brain
            probs = softmax(self.SAValues(state))
            # Set probability of illegal choices to 0
            probs = [p * int(isLegal) for (p, isLegal) in zip(probs, legal)]
            # Re-normalize probabilities
            probs = [p / sum(probs) for p in probs]
            # Choose a card at random based on the probabilities
            if softmaxFlag:
                actionIndex = random.choices(range(len(probs)), weights=probs)
            else:
                actionIndex = [np.argmax(probs)]

        return actionIndex[0]

    def learnFromHandScores(self, scores, gameState):
        reward = scores[self.number - 1]
        if gameState['dealer'] == (self.number - 1):
            reward += scores[2]
            dealerFlag = 1
        else:
            reward -= scores[2]
            dealerFlag = 0

        state = self.getThrowingFeatures(self.hand, self.cribThrow, dealerFlag)
        qValue = self.throwValue(state)

        update = self.alpha * (reward + np.max(self.SAValues(self.throwingState)) - qValue)

        self.throwingBrain.partial_fit([state], np.ravel([qValue + update]))
        
        if self.verbose:
            print(self.name + ": Learning from hand scores!")
            print('\tState: {}'.format(state))
            print('\tQ-value (pre): {}'.format(qValue))
            print('\tReward: {}'.format(reward))
            print('\tUpdate Value : {}'.format(update))
            
        if self.saveFlag:
            self.backup()

    def learnFromPegging(self, gameState):
        # If previous action is -1, we haven't had a turn yet. Nothing to learn.
        if self.prevAction < 4:
            state = self.getCurrentState(gameState)
            reward = self.getRelativeScore(gameState) - self.prevScore
            QValues = self.SAValues(self.prevState)[0]
            # Choose action for the current state
            currentMaxAction = self.chooseAction(gameState, False)
            update = self.alpha * (reward +
                                   self.gamma * self.SAValue(state, currentMaxAction) -
                                   self.SAValue(self.prevState, self.prevAction))
            QValues[self.prevAction] = QValues[self.prevAction] + update

            if self.verbose:
                print(self.name + ": Learning from pegging!")
                print('\tCurrent Maximum Value Action: {}'.format(currentMaxAction))
                print('\tReward: {}'.format(reward))
                # print('\tUpdate Value : {}'.format(update))
                temp = self.SAValues(self.prevState)
                print('\tQ Values for prev state before training: {}'.format(temp))
                print('\tTraining Values: {}'.format(QValues))

            try:
                self.peggingBrain.partial_fit([self.prevState], [QValues])
            except ValueError:
                print('Brainfreeze!')

            if self.verbose:
                print('\tQ Values for prev state after training: {}'.format(self.SAValues(self.prevState)))
                print('\tAction index: {}'.format(self.prevAction))
                print('\tDelta: {}'.format(self.SAValues(self.prevState) - temp))

    def SAValue(self, state, actionIndex):
        return self.SAValues(state)[0][actionIndex]

    def SAValues(self, state):
        actionValues = [[0, 0, 0, 0, 0]]
        try:
            actionValues = self.peggingBrain.predict([state])
        except skl.exceptions.NotFittedError:
            # If the brain hasn't seen any training data yet, will return the NotFittedError.
            actionValues = [[0, 0, 0, 0, 0]]
        finally:
            return actionValues

    def getCurrentState(self, gameState):
        handstate = [card.rank.value for card in sorted(self.playhand)]
        while len(handstate) < 4:
            handstate.append(0)

        cribstate = [card.rank.value for card in self.cribThrow]
        playstate = [card.rank.value for card in gameState['playorder']]
        while len(playstate) < 8:
            playstate.append(0)

        state = handstate + playstate + cribstate + [gameState['starter'].value()]
        state = [val / 13 for val in state]
        state.append(gameState['count'] / 31)
        state.append(int(gameState['dealer'] == (self.number - 1)))
        return state

    def throwValue(self, state):
        try:
            value = self.throwingBrain.predict([state])
        except skl.exceptions.NotFittedError:
            value = 0
        finally:
            return value

    def getThrowingFeatures(self, handCards, thrownCards, dealerFlag):
        throwingFeatures = []

        handCards.sort()
        for card in handCards:
            throwingFeatures.append(card.rank.value / 13)
            suit = [0, 0, 0, 0]
            suit[card.suit.value - 1] = 1
            throwingFeatures.extend(suit)

        thrownCards.sort()
        for card in thrownCards:
            throwingFeatures.append(card.rank.value / 13)
            suit = [0, 0, 0, 0]
            suit[card.suit.value - 1] = 1
            throwingFeatures.extend(suit)

        throwingFeatures.append(dealerFlag)

        return throwingFeatures

    def show(self):
        print('Hand:' + cardsString(sorted(self.playhand)))
        print('Crib throw:' + cardsString(self.cribThrow))

    def reset(self, gameState=None):
        super().reset()
        self.cribThrow = []


if __name__ == '__main__':
    # Initialize variables
    player1 = DeepPeg(1,False,False)
    player2 = Myrmidon(2,5,False)
    numHands = 50000
    repeatFlag = False
    windowSize = 250
        
    # Create and run arena
    arena = Arena([player1, player2],repeatFlag)
    results = arena.playHands(numHands)
    
    # Plot results from arena
    x = np.arange(1,numHands+1-windowSize,1)
    y0 = np.zeros(len(results[0])-windowSize)
    avgResult0 = np.average(results[0])
    mu0 = np.zeros(len(y0))
    y1 = np.zeros(len(results[1])-windowSize)
    avgResult1 = np.average(results[1])
    mu1 = np.zeros(len(y1))
    y2 = np.zeros(len(results[2])-windowSize)
    avgResult2 = np.average(results[2])
    mu2 = np.zeros(len(y2))
    
    for i in range(len(x)):
        y0[i] = np.average(results[0][i:i+windowSize])
        mu0[i] = np.average(avgResult0)
        y1[i] = np.average(results[1][i:i+windowSize])
        mu1[i] = np.average(avgResult1)
        y2[i] = np.average(results[2][i:i+windowSize])
        mu2[i] = np.average(avgResult2)
    
    fig, (ax0,ax1,ax2) = plt.subplots(3, 1, sharex='col')
    fig.set_size_inches(7, 6.5)
    
    moveAvg, = ax0.plot(x,y0,label='Moving Average')
    fullAvg, = ax0.plot(x,mu0,label='Trial Average\n({0:2f} points)'.format(avgResult0))
    ax0.set(ylabel='Pegging Differential', title="DeepPeg vs. Myrmidon (5 Simulations)\n(Moving Average Window Size = {0})".format(windowSize))
    ax0.grid()
    ax0.legend(handles=[moveAvg,fullAvg],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    moveAvg, = ax1.plot(x,y1,label='Moving Average')
    fullAvg, = ax1.plot(x,mu1,label='Trial Average\n({0:2f} points)'.format(avgResult1))
    ax1.set(ylabel='Hand Differential')
    ax1.grid()
    ax1.legend(handles=[moveAvg,fullAvg],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    moveAvg, = ax2.plot(x,y2,label='Moving Average')
    fullAvg, = ax2.plot(x,mu2,label='Trial Average\n({0:2f} points)'.format(avgResult2))
    ax2.set(xlabel='Hand Number', ylabel='Total Differential')
    ax2.grid()
    ax2.legend(handles=[moveAvg,fullAvg],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()

    fig.savefig("deeppegLearningCurve.png")
    plt.show()
