#!/usr/bin/env python3

################################################################################
#
# File : Monty2.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : A second player that uses first visit Monte Carlo to learn the
#               Q values for different states. A minor modification of QLearner.
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
#    - Arena.py (in local project)
#    - Deck.py (in local project)
#    - Player.py (in local project)
#    - Myrmidon.py (in local project)       * - for __name__ = '__main__' only
#    - Utilities.py (in local project)
#    - numpy (standard python library)
#    - sklearn (standard python library)
#    - os (standard python library)
#    - joblib (standard python library)
#    - warnings (standard python library)
#    - itertools (standard python library)
#    - matplotlib (standard python library) * - for __name__ = '__main__' only
#
################################################################################

# Cribbage imports
from Arena import Arena
from Deck import Card

# Player imports
from Player import Player
from Myrmidon import Myrmidon

# Utility imports
from Utilities import *
import numpy as np
import sklearn as skl
from sklearn import neural_network as sknn
import os
import joblib
import warnings
from itertools import combinations
import matplotlib.pyplot as plt

class Monty2(Player):
    def __init__(self, number, verbose=False, hiddenlayers=[50, 30, 15], alpha=0.01, epsilon=0.05, gamma=0.8,
                 filename='CarlosMonteros'):
        super().__init__(number, verbose)
        self.alpha = alpha
        # Note: epislon unused since we use softmax decision making
        self.epsilon = epsilon
        self.gamma = gamma
        self.name = 'Carlos Monteros'
        self.hiddenlayers = hiddenlayers
        self.episodeStates = []
        self.episodeActions = []
        self.episodeReturns = []
        self.goActionIndex = 14
        self.noActionIndex = 15
        self.currentAction = self.noActionIndex
        # Need to track previous score to calculate rewards
        self.prevScore = 0
        self.cribThrow = []
        self.throwingState = []
        self.filename = filename
        self.filedir = 'BrainsInJars'
        self.fullfilepegging = os.path.join(os.getcwd(), self.filedir, self.filename+"_peg.brain")
        self.fullfilethrowing = os.path.join(os.getcwd(), self.filedir, self.filename+"_throw.brain")
        if os.path.exists(self.fullfilepegging):
            self.pegbrain = joblib.load(self.fullfilepegging)
        else:
            self.pegbrain = sknn.MLPRegressor(hidden_layer_sizes=self.hiddenlayers, activation='relu', solver='adam',
                                              alpha=self.alpha, batch_size=4, max_iter=200)

        if os.path.exists(self.fullfilethrowing):
            self.throwingBrain = joblib.load(self.fullfilethrowing)
        else:
            self.throwingBrain = sknn.MLPRegressor(hidden_layer_sizes=(25, 10), activation='relu', solver='adam',
                                                   alpha=self.alpha, batch_size=1, max_iter=200)
        warnings.filterwarnings("ignore", category=UserWarning)

    def backup(self):
        if not os.path.isdir(self.filedir):
            os.mkdir(self.filedir)
        joblib.dump(self.pegbrain, self.fullfilepegging)
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
                cribCards = [thrownCards.pop(), thrownCards.pop()]

        for i in range(len(cribCards)):
            for j in range(len(self.hand)):
                if cribCards[i].isIdentical(self.hand[j]):
                    self.cribThrow.append(Card(self.hand[j].rank, self.hand[j].suit))
                    self.hand.pop(j)
                    break

        if self.verbose:
            print("{} threw {} cards into the crib".format(self.getName(), numCards))

        super().createPlayHand()

        return cribCards

    def playCard(self, gameState):
        if len(self.playhand) == 4:
            self.throwingState = self.getCurrentState(gameState)
        # record current state and score
        self.currentAction = self.chooseAction(gameState)
        if self.currentAction == self.goActionIndex:
            # Go
            cardPlayed = None
        else:
            for i in range(len(self.playhand)):
                if self.playhand[i].rank.value == self.currentAction:
                    cardPlayed = self.playhand.pop(i)
                    break
            self.episodeStates.append(self.getCurrentState(gameState))
            self.prevScore = self.getRelativeScore(gameState)
            self.episodeActions.append(self.currentAction)

        return cardPlayed

    # Explain why certain cards were thrown into the crib
    def explainThrow(self,numCards,gameState):
        print("Carlos Monteros ({}) has not implemented explainThrow".format(self.number))
        
    # Explain why a certain card was played during pegging
    def explainPlay(self,numCards,gameState):
        print("Carlos Monteros ({}) has not implemented explainThrow".format(self.number))

    def chooseAction(self, gameState):
        # Make sure hand is sorted
        self.playhand.sort()
        # Figure out which actions are legal
        legal = [(gameState['count'] + card.value()) <= 31 for card in self.playhand]
        legalHand = [card for (card, isLegal) in zip(self.playhand, legal) if isLegal]
        # legal hand is a new hand of only the cards that can legally be played
        # If no legal cards, only option is Go.
        if not legalHand:
            actionIndex = self.goActionIndex
        else:
            # Choosing a card from the legal cards
            state = self.getCurrentState(gameState)
            # Predict using brain
            probs = epsilonsoft(self.SAValues(state), self.epsilon)
            # Set probability of illegal choices to 0
            try:
                probs = [probs[card.rank.value - 1] for card in legalHand]
            except IndexError:
                print('saywhat?!')

            # Re-normalize probabilities
            probs = [p / sum(probs) for p in probs]
            # Choose a card at random based on the probabilities
            actionIndex = random.choices(range(len(probs)), weights=probs)
            actionIndex = legalHand[actionIndex[0]].rank.value

        return actionIndex

    def learnFromCutCard(self, gameState):
        pass

    def learnFromPegging(self, gameState):
        if self.currentAction == self.goActionIndex:
            self.episodeReturns[-1] += self.getRelativeScore(gameState) - self.prevScore
            self.prevScore = self.getRelativeScore(gameState)
        elif self.currentAction < self.goActionIndex:
            reward = self.getRelativeScore(gameState) - self.prevScore
            self.prevScore = self.getRelativeScore(gameState)
            self.episodeReturns.append(reward)

    def go(self, gameState):
        self.episodeReturns[-1] += self.getRelativeScore(gameState) - self.prevScore
        self.prevScore = self.getRelativeScore(gameState)

    def endOfGame(self, gameState):
        if self.currentAction == self.goActionIndex:
            self.episodeReturns[-1] += self.getRelativeScore(gameState) - self.prevScore
            self.prevScore = self.getRelativeScore(gameState)
        else:
            self.episodeReturns.append(self.getRelativeScore(gameState) - self.prevScore)
            self.prevScore = self.getRelativeScore(gameState)

    def updateBrain(self, gameState):
        if self.episodeActions:
            if len(self.episodeActions) > len(self.episodeReturns):
                self.episodeReturns.append(self.getRelativeScore(gameState) - self.prevScore)
                self.prevScore = self.getRelativeScore(gameState)
            X = [state for state in self.episodeStates]
            Y = [self.SAValues(state) for state in self.episodeStates]
            Gt = [None] * len(self.episodeStates)
            G = 0
            for i in reversed(range(len(self.episodeStates))):
                try:
                    Gt[i] = (self.gamma * G + self.episodeReturns[i])
                except IndexError:
                    print("what is happening here?")
                G = Gt[i]
                Y[i][self.episodeActions[i] - 1] = Gt[i]

            self.pegbrain.partial_fit(X, Y)

    def SAValue(self, state, actionIndex):
        return self.SAValues(state)[0][actionIndex]

    def SAValues(self, state):
        actionValues = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        try:
            if isinstance(state[1], list):
                actionValues = self.pegbrain.predict(state)
            else:
                actionValues = self.pegbrain.predict([state])
                actionValues = actionValues[0]
        except skl.exceptions.NotFittedError:
            # If the brain hasn't seen any training data yet, will return the NotFittedError.
            actionValues = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        state.append(int(gameState['dealer'] == self.number))
        return state

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

        #self.backup()

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
        self.updateBrain(gameState)
        super().reset()
        self.cribThrow = []
        self.episodeStates = []
        self.episodeActions = []
        self.episodeReturns = []
        self.currentAction = self.noActionIndex


if __name__ == "__main__":
    # Initialize variables
    player1 = Monty2(1, False, alpha=0.9, epsilon=0.1, gamma=1)
    player2 = Myrmidon(2, 5, False)
    numHands = 5000
    repeatFlag = False
    windowSize = 100

    # Create and run arena
    arena = Arena([player1, player2], repeatFlag)
    results = arena.playHands(numHands)

    # Plot results from arena
    x = np.arange(1, numHands + 1 - windowSize, 1)
    y0 = np.zeros(len(results[0]) - windowSize)
    avgResult0 = np.average(results[0])
    mu0 = np.zeros(len(y0))
    y1 = np.zeros(len(results[1]) - windowSize)
    avgResult1 = np.average(results[1])
    mu1 = np.zeros(len(y1))
    y2 = np.zeros(len(results[2]) - windowSize)
    avgResult2 = np.average(results[2])
    mu2 = np.zeros(len(y2))

    for i in range(len(x)):
        y0[i] = np.average(results[0][i:i + windowSize])
        mu0[i] = np.average(avgResult0)
        y1[i] = np.average(results[1][i:i + windowSize])
        mu1[i] = np.average(avgResult1)
        y2[i] = np.average(results[2][i:i + windowSize])
        mu2[i] = np.average(avgResult2)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex='col')
    fig.set_size_inches(7, 6.5)

    moveAvg, = ax0.plot(x, y0, label='Moving Average')
    fullAvg, = ax0.plot(x, mu0, label='Trial Average\n({0:2f} points)'.format(avgResult0))
    ax0.set(ylabel='Pegging Differential',
            title="Carlos Monteros ("
                  + u"\u03B1" + " = {}, ".format(player1.alpha)
                  + u"\u03B5" + " = {}, ".format(player1.epsilon)
                  + u"\u03B3" + " = {1}) \n vs. Myrmidon (5 Simulations)\n(Moving Average Window Size = {0})".format(
                windowSize, player1.gamma))
    ax0.grid()
    ax0.legend(handles=[moveAvg, fullAvg], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    moveAvg, = ax1.plot(x, y1, label='Moving Average')
    fullAvg, = ax1.plot(x, mu1, label='Trial Average\n({0:2f} points)'.format(avgResult1))
    ax1.set(ylabel='Hand Differential')
    ax1.grid()
    ax1.legend(handles=[moveAvg, fullAvg], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    moveAvg, = ax2.plot(x, y2, label='Moving Average')
    fullAvg, = ax2.plot(x, mu2, label='Trial Average\n({0:2f} points)'.format(avgResult2))
    ax2.set(xlabel='Hand Number', ylabel='Total Differential')
    ax2.grid()
    ax2.legend(handles=[moveAvg, fullAvg], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    fig.savefig("CarlosMonteros_a" + str(player1.alpha) +
                "_e" + str(player1.epsilon) +
                "_g" + str(player1.gamma) +
                "_LearningCurve_NonStationary.png")
    plt.show()
    player1.backup()