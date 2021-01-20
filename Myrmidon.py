#!/usr/bin/env python3

################################################################################
#
# File : Myrmidon.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : A Player that makes use of one-step rollouts and heuristics.
#
# Notes : If this file is run, it instantiates an Arena and measures this agent's
#         performance against itself.
#
# Dependencies:
#    - Utilities.py (in local project)
#    - Deck.py (in local project)
#    - Scoring.py (in local project)
#    - Arena.py (in local project)           * - for __name__ = '__main__' only
#    - Player.py (in local project)
#    - numpy (standard python library)
#    - itertools (standard python library)
#    - random (standard python library)
#    - matplotlib (standard python library)  * - for __name__ = '__main__' only
#
################################################################################

# Cribbage imports
from Utilities import *
from Deck import Card
from Scoring import *
from Arena import Arena

# Player imports
from Player import Player

# Utility imports
import numpy as np
from itertools import combinations
import random
import matplotlib.pyplot as plt

class Myrmidon(Player):

    def __init__(self, number, numSims,verboseFlag):
        super().__init__(number)
        self.numSims = max(numSims,1)
        self.verbose = verboseFlag
        self.name = "Myrmidon"
        self.cribThrow = []

    def reset(self, gameState=None):
        super().reset()

    # If the argument card is in the argument set of cards, return true.
    # Otherwise return false.
    def inHand(self, cardToCheck):
        return cardToCheck in self.hand

    # Returns a random starter card that isn't in the player's hand
    def randomStarter(self):
        newCard = Card(random.randint(1, 13), random.randint(1, 4))

        while newCard in self.hand:
            newCard = Card(random.randint(1, 13), random.randint(1, 4))

        return newCard

    # Chooses which cards to throw based on randomly sampling potential starter
    # cards for each combination of cards thrown.
    def throwCribCards(self, numCards, gameState, criticThrow):
        cribCards = []
        cardScores = np.zeros(len(self.hand))

        # Score the cards that would be left in the player's hand
        for combination in combinations(self.hand, len(self.hand) - numCards):
            for i in range(0, self.numSims):
                starterCard = self.randomStarter()
                score = getScore(list(combination), starterCard, False)
                for j in range(0, len(self.hand)):
                    if self.hand[j] in combination:
                        cardScores[j] += score

        # Score the cards that would be thrown in the crib
        for combination in combinations(self.hand, numCards):
            for i in range(0, self.numSims):
                starterCard = self.randomStarter()
                score = getScore(list(combination), starterCard, False)
                for j in range(0, len(self.hand)):
                    if self.hand[j] in combination:
                        if gameState['dealer'] == (self.number - 1):
                            # We can worry less about keeping the card if it
                            # will score points for us in the crib
                            cardScores[j] -= score
                            if self.hand[j].rank == 5:
                                cardScores[j] += 2
                        else:
                            # We should keep cards that will score points for 
                            # our opponents in the crib
                            cardScores[j] += score

        # Pick the lowest scoring cards to throw
        for i in range(0, numCards):
            lowIndex = min(range(len(cardScores)), key=cardScores.__getitem__)
            cribCards.append(self.hand.pop(lowIndex))
            cardScores = np.delete(cardScores, lowIndex)

        if not(criticThrow is None):
            if not(areCardsEqual(cribCards,criticThrow)):
                self.explainThrow(numCards,gameState)
            else:
                print("The critic agreed with {}'s throw.".format(self.getName()))

        if(self.verbose):
            print("Myrmidon ({}) threw {} cards into the crib".format(self.number, numCards))

        super().createPlayHand()

        self.cribThrow = cribCards

        return cribCards

    def explainThrow(self,numCards,gameState):
        hand = []
        for i in range(len(self.hand)):
            hand.append(self.hand[i])
        for i in range(numCards):
            hand.append(self.cribThrow[i])
            
        cardScores = np.zeros(len(hand))
        if gameState['dealer'] == self.number - 1:
            dealerFlag = 1
        else:
            dealerFlag = 0    
        
        print("Myrmidon ({}) is considering a hand of: {}".format(self.number,cardsString(hand)),end="")        
        if dealerFlag == 1:
            print(". Own crib.")
        else:
            print(". Opponent's crib.")
        
        # Score the cards that would be left in the player's hand
        for combination in combinations(hand, len(hand) - numCards):
            for i in range(0, self.numSims):
                starterCard = self.randomStarter()
                score = getScore(list(combination), starterCard, False)
                for j in range(0, len(hand)):
                    if hand[j] in combination:
                        cardScores[j] += score

        # Score the cards that would be thrown in the crib
        for combination in combinations(self.hand, numCards):
            for i in range(0, self.numSims):
                starterCard = self.randomStarter()
                score = getScore(list(combination), starterCard, False)
                for j in range(0, len(hand)):
                    if hand[j] in combination:
                        if gameState['dealer'] == (self.number - 1):
                            # We can worry less about keeping the card if it
                            # will score points for us in the crib
                            cardScores[j] -= score
                            if hand[j].rank == 5:
                                cardScores[j] += 2
                        else:
                            # We should keep cards that will score points for 
                            # our opponents in the crib
                            cardScores[j] += score
        
        for i in range(len(hand)):
            print("{}: {}".format(str(hand[i]),cardScores[i]))
        
        # Pick the lowest scoring cards to throw
        cribCards = []
        for i in range(0, numCards):
            lowIndex = min(range(len(cardScores)), key=cardScores.__getitem__)
            cribCards.append(hand[lowIndex])
            cardScores = np.delete(cardScores, lowIndex)
            
        print("I chose to throw: {}".format(cardsString(cribCards)))
        print("")

    # Chooses a card to play during pegging by maximizing the immediate return
    # and the value of the afterstate according to some heuristic rules.
    def playCard(self, gameState, criticCard):
        cardScores = np.zeros(len(self.playhand))
        playedCard = None
        countCards = gameState['inplay']
        count = gameState['count']

        if len(self.playhand) != 0:
            for i in range(0, len(self.playhand)):
                # Check that the card can be played
                if count + self.playhand[i].value() < 32:
                    newCountCards = countCards + [self.playhand[i]]
                    cardScores[i] += 10 * scoreCards(newCountCards, False) + self.playhand[i].rank.value
                    if (count + self.playhand[i].value() == 5) or (count + self.playhand[i].value() == 10) or (
                            count + self.playhand[i].value() == 21):
                        cardScores[i] = max(1, cardScores[i] - 10)
                    if count + self.playhand[i].value() < 5:
                        cardScores[i] += 15

            if np.amax(cardScores) > 0:
                if not(criticCard is None) and not(criticCard.isIdentical(self.playhand[max(range(len(cardScores)))])):
                    self.explainPlay(gameState)
                playedCard = self.playhand.pop(max(range(len(cardScores)), key=cardScores.__getitem__))
                if(self.verbose):
                    print("\tMyrmidon ({}) played {}".format(self.number, str(playedCard)))
            else:
                if(self.verbose):
                    print("\tMyrmidon ({}) says go!".format(self.number))
        else:
            if(self.verbose):
                print("\tMyrmidon ({}) has no cards left; go!".format(self.number))

        return playedCard

    def explainPlay(self,gameState):
        cardScores = np.zeros(len(self.playhand))
        playedCard = None
        countCards = gameState['inplay']
        count = gameState['count']

        if len(self.playhand) != 0:
            print("\tMyrmidon ({}) is considering:".format(self.number))
            for i in range(0, len(self.playhand)):
                print("\t{}".format(str(self.playhand[i])),end=" ")
                # Check that the card can be played
                if count + self.playhand[i].value() < 32:
                    newCountCards = countCards + [self.playhand[i]]
                    cardScores[i] += 10 * scoreCards(newCountCards, False) + self.playhand[i].rank.value
                    print("scores {} for the count,".format(cardScores[i]),end=" ")
                    if (count + self.playhand[i].value() == 5) or (count + self.playhand[i].value() == 10) or (
                            count + self.playhand[i].value() == 21):
                        cardScores[i] = max(1, cardScores[i] - 10)
                        print("is adjusted to {0} for the count left ({1}),".format(cardScores[i],(count + self.playhand[i].value())),end=" ")
                    if count + self.playhand[i].value() < 5:
                        cardScores[i] += 15
                        print("is rewarded for leaving a count of less than 5.",end=" ")
                    print("Final score is {}.".format(cardScores[i]))
                else:
                    print("can't be played.")
            if np.amax(cardScores) > 0:
                playedCard = self.playhand[max(range(len(cardScores)))]
                print("\tI choose to play {}".format(str(playedCard)))
            else:
                print("\tI can't play any of my cards and have to say go!".format(self.number))
        else:
            print("\tMyrmidon ({}): I have no cards left and have to say go!".format(self.number))

    # Myrmidon does not learn
    def learnFromHandScores(self, scores, gameState):
        pass
    
    # Myrmidon does not learn
    def learnFromPegging(self, gameState):
        pass

    def show(self):
        print('{}:'.format(self.getName()))
        print('Hand:' + cardsString(sorted(self.playhand)))

if __name__ == '__main__':
    # Initialize variables
    player1 = Myrmidon(1,5,False)
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
    ax0.set(ylabel='Pegging Differential', title="Myrmidon (5 Simulations) vs. Myrmidon (5 Simulations)\n(Moving Average Window Size = {0})".format(windowSize))
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

    fig.savefig("myrmidon_ns05_LearningCurveNonStationary.png")
    plt.show()