#!/usr/bin/env python3

################################################################################
#
# File : Player.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : An abstract class defining what methods a player class must have
#               in order to play well with Cribbage.py.
#
# Notes : Abstract methods are: throwCribCards, playCard, learnFromHandScores,
#         and learnFromPegging
#
#         The verboseFlag is used to control whether or not the print commands
#         are used throughout the file. 
#
# Dependencies:
#    - Deck.py (in local project)
#    - abc (standard python library)
#
################################################################################

from abc import ABC, abstractmethod
from Deck import Card

class Player(ABC):
    def __init__(self, number, verbose=False):
        self.hand = []
        self.playhand = []
        self.number = number
        self.pips = 0
        self.name = "Generic Player"
        self.verbose = verbose

    def newGame(self, gameState):
        self.reset(gameState)
        self.pips = 0

    def reset(self, gameState=None):
        self.hand = []
        self.playhand = []

    def removeCard(self, card):
        for i in range(0,len(self.playhand)):
            if self.playhand[i].isIdentical(card):
                self.playhand.pop(i)
                return
        print("Tried to remove the {} from {}'s playhand, but it wasn't there!".format(str(card),self.getName()))

    @abstractmethod
    def throwCribCards(self, numCards, crib, criticThrow):
        pass

    @abstractmethod
    def playCard(self, gameState, criticCard):
        pass

    @abstractmethod
    def explainThrow(self, numCards, crib):
        pass
    
    @abstractmethod
    def explainPlay(self, gameState):
        pass

    @abstractmethod
    def learnFromHandScores(self, scores, gameState):
        pass
    
    @abstractmethod
    def learnFromPegging(self, gameState):
        pass

    def endOfGame(self, gameState):
        pass

    def thirtyOne(self, gameState):
        pass

    def go(self, gameState):
        pass

    def createPlayHand(self):
        for i in range(0, len(self.hand)):
            self.playhand.append(Card(self.hand[i].rank, self.hand[i].suit))

    def getRelativeScore(self, gameState):
        score = 0
        try:
            if self.number == 1:
                score = gameState['scores'][0] - gameState['scores'][1]
            else:
                score = gameState['scores'][1] - gameState['scores'][0]
        except:
            print('Problems')
        finally:
            return score

    def getName(self):
        return str(self.name + "({0})".format(self.number))

    def __str__(self):
        hand = "{}: ".format(self.getName())
        for card in self.hand:
            hand = hand + str(card) + ", "

        hand = hand[:-2]
        return hand

    def show(self):
        print("{} has scored {} pips.".format(self.getName(), self.pips))
        if self.hand:
            print("Current hand is:")
            for card in self.hand:
                print("\t{}".format(str(card)))
        else:
            print("Current hand is empty.")

    def isInHand(self, checkCard):
        return checkCard in self.hand

    def isInPlayHand(self, checkCard):
        return checkCard in self.playhand

    def draw(self, deck):
        self.hand.append(deck.cards.pop())
