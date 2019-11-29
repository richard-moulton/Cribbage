#!/usr/bin/env python3

################################################################################
#
# File : LinearB.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : A Player that represents hands using a linear combination of 
#               features. These features are then used for episodic semi-gradient
#               one-step Sarsa during the throwing cards phase and for true 
#               online Sarsa during the pegging phase.
#               
# Notes : If this file is run, it instantiates an Arena and measures this agent's
#         performance against Myrmidon.
#
# Dependencies:
#    - Deck.py (in local project)
#    - Arena.py (in local project)          * - for __name__ = '__main__' only
#    - Player.py (in local project)
#    - Myrmidon.py (in local project)       * - for __name__ = '__main__' only
#    - Utilities.py (in local project)
#    - numpy (standard python library)
#    - itertools (standard python library)
#    - pathlib (standard python library)
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
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt

class LinearB(Player):

    def __init__(self, number, alpha, Lambda, verboseFlag):
        super().__init__(number)

        # Set learning parameters
        self.stepSize = alpha
        self.traceDecay = Lambda
        self.verbose = verboseFlag
        self.name = "LinearB"

        # (Attempt to) Load weight arrays
        self.throwingWeights = np.zeros(9)
        self.peggingWeights = np.zeros(7)
        throwWeightsFile = Path("./throwWeights.npy")
        pegWeightsFile = Path("./pegWeights.npy")

        try:
            absThrowWeights = throwWeightsFile.resolve(strict=True)
            self.throwingWeights = np.load(absThrowWeights)
        except FileNotFoundError:
            print("./throwWeights.npy was not found, starting with weights initialized to zeros.")
            self.throwingWeights = np.zeros(9)
        finally:
            pass

        try:
            absPegWeights = pegWeightsFile.resolve(strict=True)
            self.peggingWeights = np.load(absPegWeights)
        except FileNotFoundError:
            print("./pegWeights.npy was not found, starting with weights initialized to zeros.")
            self.peggingWeights = np.zeros(7)
        finally:
            pass

        # Initialize some global variables
        self.peggingMemory = np.zeros(7)
        self.cardsThrown = []
        self.state = []
        self.score = 0
        self.qOld = 0

    def reset(self, gameState=None):
        super().reset()

    # Convert a state into a feature vector for throwing cards into the crib
    def getFeaturesThrowing(self, handCards, thrownCards, crib):
        lowCards = sum(1 for c in handCards if c.value() < 5) / 4
        fives = sum(1 for c in handCards if c.value() == 5) / 4
        highCards = sum(1 for c in handCards if (c.value() > 5 and c.value() < 10)) / 4
        tens = sum(1 for c in handCards if c.value() == 10) / 4
        lowCardsThrown = sum(1 for c in thrownCards if c.value() < 5) / 2
        fivesThrown = sum(1 for c in thrownCards if c.value() == 5) / 2
        highCardsThrown = sum(1 for c in thrownCards if (c.value() > 5 and c.value() < 10)) / 2
        tensThrown = sum(1 for c in thrownCards if c.value() == 10) / 2
        if crib == (self.number - 1):
            dealer = 1
        else:
            dealer = 0

        return np.array(
            [lowCards, fives, highCards, tens, lowCardsThrown, fivesThrown, highCardsThrown, tensThrown, dealer])

    # Convert a state into a feature vector for pegging
    def getFeaturesPegging(self, cards, count, oppCards):
        lowCards = sum(1 for c in cards if c.value() < 5) / 4
        fives = sum(1 for c in cards if c.value() == 5) / 4
        highCards = sum(1 for c in cards if (c.value() > 5 and c.value() < 10)) / 4
        tens = sum(1 for c in cards if c.value() == 10) / 4
        oppCards = oppCards / 4

        count = count / 31
        runPossible = 0

        if (lowCards + fives + highCards + tens + oppCards == 0):
            count = 0

        if (len(cards) > 1) and (np.abs(cards[len(cards) - 1].rank.value - cards[len(cards) - 2].rank.value) == 1):
            runPossible = 1

        return np.array([lowCards, fives, highCards, tens, oppCards, count, runPossible])

    # Get the relative score between this player and their opponent
    # implemented in Player Class

    # Choose the specified number of cards to throw into the crib
    def throwCribCards(self,numCards,gameState):
        self.cardsThrown = []
        cribCards = []
        cardIndices = list(range(len(self.hand)))
        maxValue = -np.inf

        for combination in combinations(cardIndices, len(self.hand) - numCards):
            handCards = []
            thrownCards = []
            for i in range(0, len(cardIndices)):
                if i in combination:
                    handCards.append(Card(self.hand[i].rank, self.hand[i].suit))
                else:
                    thrownCards.append(Card(self.hand[i].rank, self.hand[i].suit))

            q = np.matmul(np.transpose(self.throwingWeights),self.getFeaturesThrowing(handCards,thrownCards,gameState['dealer']))

            if q > maxValue:
                maxValue = q
                cribCards = []
                cribCards.append(thrownCards.pop())
                cribCards.append(thrownCards.pop())

        for i in range(0, len(cribCards)):
            for j in range(0, len(self.hand)):
                if cribCards[i].isIdentical(self.hand[j]):
                    self.cardsThrown.append(Card(self.hand[j].rank, self.hand[j].suit))
                    self.hand.pop(j)
                    break

        if (self.verbose):
            print("{} threw {} into the crib".format(self.getName(), cardsString(self.cardsThrown)))

        super().createPlayHand()

        return cribCards;

    # Choose which card to play during the pegging phase
    def playCard(self, gameState):
        # Initialize for this episode
        if len(self.playhand) == 4:
            self.throwingMemory = np.zeros(8);

        # Initialize for this decision
        playedCard = None

        if len(self.playhand) != 0:
            playedCard = self.selectCardToPlay(gameState)
            if playedCard is None:
                if (self.verbose):
                    print("\t{} says go!".format(self.getName()))
            else:
                index = -1
                for i in range(0, len(self.playhand)):
                    if self.playhand[i].isIdentical(playedCard):
                        index = i
                        break
                self.playhand.pop(index)
                newCount = gameState['count'] + playedCard.value()
                self.state = self.getFeaturesPegging(self.playhand, newCount, gameState['numCards'][self.number % 2])
                self.score = self.getRelativeScore(gameState)
                if (self.verbose):
                    print("\t{} played {}".format(self.getName(), str(playedCard)))
        else:
            if (self.verbose):
                print("\t{} has no cards left; go!".format(self.getName()))

        return playedCard

    # Determine which specific card should be played in the argument state
    def selectCardToPlay(self, gameState):
        cardToPlay = None
        cardScores = np.zeros(len(self.playhand))

        # Select A according to w and x
        if len(self.playhand) != 0:
            for i in range(0, len(self.playhand)):
                # Check that the card can be played
                if (gameState['count'] + self.playhand[i].value() < 32):
                    newHand = []
                    for i in range(0, len(self.playhand)):
                        newHand.append(Card(self.playhand[i].rank, self.playhand[i].suit))
                    newHand.pop(i)
                    newCount = gameState['count'] + self.playhand[i].value()
                    cardScores[i] = np.matmul(self.peggingWeights, self.getFeaturesPegging(newHand, newCount,
                                                                                           gameState['numCards'][
                                                                                               self.number % 2]))
                else:
                    cardScores[i] = -np.Inf

            if not (np.isinf(np.amax(cardScores))):
                cardToPlay = self.playhand[max(range(len(cardScores)), key=cardScores.__getitem__)]

        return cardToPlay

    # Useful for diagnostics
    def arrayToString(self, array):
        stringArray = []
        for i in range(0, len(array)):
            stringArray.append(str(array[i]))

        return stringArray

    # Episodic semi-gradient Sarsa from Sutton and Barto, p. 244
    def learnFromHandScores(self, scores, gameState):
        state = self.getFeaturesThrowing(self.hand, self.cardsThrown, gameState['dealer'])
        reward = scores[self.number-1] + (np.matmul(self.peggingWeights, self.getFeaturesPegging(self.hand, 0, 4)))
        if gameState['dealer'] == self.number - 1:
            reward += scores[2]
        else:
            reward -= scores[2]
        
        self.throwingWeights = self.throwingWeights + (
                self.stepSize * (reward - np.matmul(self.throwingWeights, state)) * state)

        np.save('./throwWeights.npy', self.throwingWeights)

    # True Online Sarsa(lambda) from Sutton and Barto, p. 307
    def learnFromPegging(self, gameState):
        if len(self.state) > 0:
            # Choose A' greedily from S' using w
            nextCard = self.selectCardToPlay(gameState)
            nextCount = gameState['count']
            newHand = []
            for card in self.playhand:
                newHand.append(Card(card.rank, card.suit))

            if not (nextCard is None):
                for i in range(0, len(newHand)):
                    if newHand[i].isIdentical(nextCard):
                        newHand.pop(i)
                        break
                nextCount += nextCard.value()

            # Get x and x'
            x = self.state
            xPrime = self.getFeaturesPegging(newHand, nextCount, gameState['numCards'][self.number % 2])

            # Calculate Q, Q', R and delta
            q = np.matmul(self.peggingWeights, x)
            qPrime = np.matmul(self.peggingWeights, xPrime)
            R = self.getRelativeScore(gameState) - self.score
            delta = R + qPrime - q

            # Update memory and weights
            self.peggingMemory = (self.traceDecay * self.peggingMemory) + (
                    (1 - (self.stepSize * self.traceDecay * np.matmul(np.transpose(self.peggingMemory), x))) * x)
            self.peggingWeights = self.peggingWeights + (
                    self.stepSize * (delta + q - self.qOld) * self.peggingMemory) - (
                                          self.stepSize * (q - self.qOld) * x)

            # Bookkeeping to prepare for the next step
            self.qOld = qPrime

            np.save("./pegWeights.npy", self.peggingWeights)

if __name__ == '__main__':
    # Initialize variables
    player1 = LinearB(1,0.5,0.9,False)
    player2 = Myrmidon(2,5,False)
    numHands = 5000
    repeatFlag = False
    windowSize = 100
        
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
    ax0.set(ylabel='Pegging Differential', title="LinearB ("+u"\u03B1"+" = 0.5, "+u"\u03BB"+" = 0.9) vs. Myrmidon (5 Simulations)\n(Moving Average Window Size = {0})".format(windowSize))
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

    fig.savefig("linearb_a05l09_LearningCurveNonStationary.png")
    plt.show()