#!/usr/bin/env python3

################################################################################
#
# File : Arena.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : Records performance data for a player over a number of hands.
#               Can be used to produce training curve data or to measure final
#               performance levels. 
#
# Notes :
#
# Dependencies:
#    - Cribbage.py (in local project)
#    - Deck.py (in local project)
#    - Utilities.py (in local project)
#    - numpy (standard python library)
#
################################################################################

# Cribbage imports
from Cribbage import Cribbage
from Deck import *

# Utility imports
import numpy as np
from Utilities import *


class Arena():
    def __init__(self, players, repeatDeck):
        # Initialize the players
        self.numPlayers = len(players)

        # Initialize the Cribbage Dojo
        self.repeatFlag = repeatDeck
        self.cribbageDojo = Cribbage(players,False,True)

        print("Beginning the Arena between {0} and {1}.".format(self.cribbageDojo.players[0].getName(),self.cribbageDojo.players[1].getName()))

    def playHands(self, numHands):
        peggingDiff = np.zeros(numHands)
        handsDiff = np.zeros(numHands)
        totalPointsDiff = np.zeros(numHands)
        
        for handNumber in range(numHands):
            if handNumber%100 == 0:
                print("Playing hand {} of {}.".format(handNumber+1,numHands))
                
            # Initialize the hand
            if self.repeatFlag:
                self.deck = RiggedDeck(1)
                self.deck.shuffle()
            else:
                self.deck = Deck(1)
                self.deck.shuffle()
            hands = []
            scores = []
            pegScores = []
            handScores = []

            # Deal two hands of four
            for i in range(0, self.numPlayers):
                hands.append([])
                scores.append(0)
                pegScores.append(0)
                handScores.append(0)
                for j in range(0, 6):
                    hands[i].append(self.deck.cards.pop())

            #print("Hand 1 is "+cardsString(hands[0]))
            #print("Hand 2 is "+cardsString(hands[1]))
            starterCard = self.deck.cards.pop()

            # Assign these hands to the players
            for i in range(len(hands[0])):
                self.cribbageDojo.players[0].hand.append(Card(hands[0][i].rank, hands[0][i].suit))
                self.cribbageDojo.players[1].hand.append(Card(hands[1][i].rank, hands[1][i].suit))

            #print(self.cribbageDojo.players[0].getName()+" has the cards "+cardsString(self.cribbageDojo.players[0].hand))
            #print(self.cribbageDojo.players[1].getName()+" has the cards "+cardsString(self.cribbageDojo.players[1].hand))

            # FIRST PLAY THROUGH OF THE HAND
            self.cribbageDojo.dealer = 0
            self.cribbageDojo.createCrib()
            self.cribbageDojo.cut(starterCard)
            #print("The starter card is cut: "+str(self.cribbageDojo.starter))
            self.cribbageDojo.play()
            for i in range(self.numPlayers):
                pegScores[i] = self.cribbageDojo.players[i].pips
            self.cribbageDojo.scoreHands()
            for i in range(self.numPlayers):
                scores[i] = self.cribbageDojo.players[i].pips
                handScores[i] = scores[i] - pegScores[i]
            self.cribbageDojo.resetGame()

            peggingDiff[handNumber] = peggingDiff[handNumber] + pegScores[0] - pegScores[1]
            handsDiff[handNumber] = handsDiff[handNumber] + handScores[0] - handScores[1]
            totalPointsDiff[handNumber] = totalPointsDiff[handNumber] + scores[0] - scores[1]

            #print("Hand Play 1 --> Peg: {0}-{1} ({2}), Hands: {3}-{4} ({5}), Total: {6}-{7} ({8})".format(pegScores[0],pegScores[1],peggingDiff[handNumber],handScores[0],handScores[1],handsDiff[handNumber],scores[0],scores[1],totalPointsDiff[handNumber]))


            # Assign the opposite hands to the players
            for i in range(len(hands[0])):
                self.cribbageDojo.players[1].hand.append(Card(hands[0][i].rank, hands[0][i].suit))
                self.cribbageDojo.players[0].hand.append(Card(hands[1][i].rank, hands[1][i].suit))
                
            for i in range(self.numPlayers):
                pegScores[i] = 0
                handScores[i] = 0

            # print(self.cribbageDojo.players[0].getName()+" has the cards "+cardsString(self.cribbageDojo.players[0].hand))
            # print(self.cribbageDojo.players[1].getName()+" has the cards "+cardsString(self.cribbageDojo.players[1].hand))


            # SECOND PLAY THROUGH OF THE HAND
            self.cribbageDojo.dealer = 1
            self.cribbageDojo.createCrib()
            self.cribbageDojo.cut(starterCard)
            #print("The starter card is cut: "+str(self.cribbageDojo.starter))
            self.cribbageDojo.play()
            for i in range(self.numPlayers):
                pegScores[i] = self.cribbageDojo.players[i].pips
            self.cribbageDojo.scoreHands()
            for i in range(self.numPlayers):
                scores[i] = self.cribbageDojo.players[i].pips
                handScores[i] = scores[i] - pegScores[i]
            self.cribbageDojo.resetGame()
            
            peggingDiff[handNumber] = peggingDiff[handNumber] + pegScores[0] - pegScores[1]
            handsDiff[handNumber] = handsDiff[handNumber] + handScores[0] - handScores[1]
            totalPointsDiff[handNumber] = totalPointsDiff[handNumber] + scores[0] - scores[1]
            
            #print("Hand Play 2 --> Peg: {0}-{1} ({2}), Hands: {3}-{4} ({5}), Total: {6}-{7} ({8})".format(pegScores[0],pegScores[1],peggingDiff[handNumber],handScores[0],handScores[1],handsDiff[handNumber],scores[0],scores[1],totalPointsDiff[handNumber]))

            
            #print("Hand {0}: Pegging Diff {1}, Hands Diff {2}, Total Diff {3}".format(handNumber+1, peggingDiff[handNumber], handsDiff[handNumber], totalPointsDiff[handNumber]))
            
        return [peggingDiff,handsDiff,totalPointsDiff]