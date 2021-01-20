#!/usr/bin/env python3

################################################################################
#
# File : Arena.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : Outputs performance data for a player over a number of hands 
#               along with commentary from a critic player.
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

class CriticSessions():
    def __init__(self, players, critic, verboseFlag):
        # Initialize the players
        self.numPlayers = len(players)
        self.critic = critic
        self.verbose = verboseFlag
        
        # Initialize the Cribbage Dojo
        self.cribbageDojo = Cribbage(players,self.critic,self.verbose,False)

        print("Beginning the Critic Sessions between {0} (with {2} critiquing) and {1}.".format(self.cribbageDojo.players[0].getName(),self.cribbageDojo.players[1].getName(),self.critic.getName()))
        
    def playHands(self, numHands):
        peggingDiff = np.zeros(numHands)
        handsDiff = np.zeros(numHands)
        totalPointsDiff = np.zeros(numHands)
        
        for handNumber in range(numHands):
            if handNumber%100 == 0:
                print("Playing hand {} of {}.".format(handNumber+1,numHands))
                
            # Initialize the hand
            self.deck = Deck(1)
            self.deck.shuffle()
            hands = []
            scores = []
            pegScores = []
            handScores = []

            # Deal two hands of six
            for i in range(0, self.numPlayers):
                hands.append([])
                scores.append(0)
                pegScores.append(0)
                handScores.append(0)
                for j in range(0, 6):
                    hands[i].append(self.deck.cards.pop())

            if self.verbose:
                print("Hand 1 is "+cardsString(hands[0]))
                print("Hand 2 is "+cardsString(hands[1]))
                
            starterCard = self.deck.cards.pop()

            # Assign these hands to the players and give a copy of the first hand to the 
            self.critic.hand.clear()
            for i in range(len(hands[0])):
                self.cribbageDojo.players[0].hand.append(Card(hands[0][i].rank, hands[0][i].suit))
                self.critic.hand.append(Card(hands[0][i].rank, hands[0][i].suit))
                self.cribbageDojo.players[1].hand.append(Card(hands[1][i].rank, hands[1][i].suit))

            if self.verbose:
                print(self.cribbageDojo.players[0].getName()+" has the cards "+cardsString(self.cribbageDojo.players[0].hand))
                print(self.cribbageDojo.players[1].getName()+" has the cards "+cardsString(self.cribbageDojo.players[1].hand))

            # FIRST PLAY THROUGH OF THE HAND
            self.cribbageDojo.dealer = 0
            self.cribbageDojo.createCrib()
            self.cribbageDojo.cut(starterCard)
            if self.verbose:
                print("The starter card is cut: "+str(self.cribbageDojo.starter))
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

            if self.verbose:
                print("Hand Play 1 --> Peg: {0}-{1} ({2}), Hands: {3}-{4} ({5}), Total: {6}-{7} ({8})\n".format(pegScores[0],pegScores[1],peggingDiff[handNumber],handScores[0],handScores[1],handsDiff[handNumber],scores[0],scores[1],totalPointsDiff[handNumber]))
                print("***********");

            # Assign the opposite hands to the players
            self.critic.hand.clear()
            for i in range(len(hands[0])):
                self.cribbageDojo.players[1].hand.append(Card(hands[0][i].rank, hands[0][i].suit))
                self.cribbageDojo.players[0].hand.append(Card(hands[1][i].rank, hands[1][i].suit))
                self.critic.hand.append(Card(hands[1][i].rank, hands[1][i].suit))
                
            for i in range(self.numPlayers):
                pegScores[i] = 0
                handScores[i] = 0

            if self.verbose:
                print(self.cribbageDojo.players[0].getName()+" has the cards "+cardsString(self.cribbageDojo.players[0].hand))
                print(self.cribbageDojo.players[1].getName()+" has the cards "+cardsString(self.cribbageDojo.players[1].hand))


            # SECOND PLAY THROUGH OF THE HAND
            self.cribbageDojo.dealer = 1
            self.cribbageDojo.createCrib()
            self.cribbageDojo.cut(starterCard)
            if self.verbose:
                print("The starter card is cut: "+str(self.cribbageDojo.starter))
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
            
            if self.verbose:
                print("Hand Play 2 --> Peg: {0}-{1} ({2}), Hands: {3}-{4} ({5}), Total: {6}-{7} ({8})".format(pegScores[0],pegScores[1],pegScores[0] - pegScores[1],handScores[0],handScores[1],handScores[0] - handScores[1],scores[0],scores[1],scores[0] - scores[1]))
                print("***********");
                print("Hand {0} Cumulative Results: Pegging Diff {1}, Hands Diff {2}, Total Diff {3}".format(handNumber+1, peggingDiff[handNumber], handsDiff[handNumber], totalPointsDiff[handNumber]))
            
        return [peggingDiff,handsDiff,totalPointsDiff]