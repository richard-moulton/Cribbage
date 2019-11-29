#!/usr/bin/env python3

################################################################################
#
# File : TrainHand.py
#
# Authors : AJ Marasco and Richard Moulton
#
# Description : Trains players on a single hand of Cribbage
#
# Notes :
#
# Dependencies:
#    - Cribbage.py (in local project)
#    - Deck.py (in local project)
#    - PlayerRandom.py (in local project)
#    - Myrmidon.py (in local project)
#    - LinearB.py (in local project)
#    - NonLinearB.py (in local project)
#    - DeepPeg.py (in local project)
#    - Monty.py (in local project)
#    - Monty2.py (in local project)
#    - numpy (standard python Library)
#
################################################################################

# Cribbage imports
from Cribbage import Cribbage
from Deck import Deck, Card

# Player imports
from PlayerRandom import PlayerRandom
from Myrmidon import Myrmidon
from LinearB import LinearB
from NonLinearB import NonLinearB
from DeepPeg import DeepPeg
from Monty import Monty
from Monty2 import Monty2

# Utility imports
import numpy as np

class TrainHand():
    def __init__(self, players):
        # Initialize the players
        self.numPlayers = len(players)

        # Initialize the Cribbage Dojo
        self.cribbageDojo = Cribbage(players, False)
        self.cribbageDojo.dealer = 0

        print("Initialized a game of cribbage with {0} players:".format(self.numPlayers))
        for i in range(0, self.numPlayers):
            print("   " + self.cribbageDojo.players[i].getName())

    def runTrials(self, numTrials):
        cumulativeScore = 0
        cumulativePegging = 0
        cumulativeHands = 0
        for trial in range(numTrials):
            # Initialize the hand
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

            # print("Hand 1 is "+cardsString(hands[0]))
            # print("Hand 2 is "+cardsString(hands[1]))
            starterCard = self.deck.cards.pop()

            # Assign these hands to the players
            for i in range(len(hands[0])):
                self.cribbageDojo.players[0].hand.append(Card(hands[0][i].rank, hands[0][i].suit))
                self.cribbageDojo.players[1].hand.append(Card(hands[1][i].rank, hands[1][i].suit))

            # print(self.cribbageDojo.players[0].getName()+" has the cards "+cardsString(self.cribbageDojo.players[0].playhand))
            # print(self.cribbageDojo.players[1].getName()+" has the cards "+cardsString(self.cribbageDojo.players[1].playhand))

            self.cribbageDojo.dealer = 0
            self.cribbageDojo.createCrib()
            self.cribbageDojo.cut(starterCard)
            #print("The starter card is cut: "+str(self.cribbageDojo.starter))
            self.cribbageDojo.play()
            for i in range(self.numPlayers):
                pegScores[i] = self.cribbageDojo.players[i].pips
            print("Pegging for this hand was {0} - {1}".format(pegScores[0], pegScores[1]))
            self.cribbageDojo.scoreHands()
            for i in range(self.numPlayers):
                handScores[i] = self.cribbageDojo.players[i].pips - pegScores[i]
                scores[i] = scores[i] + self.cribbageDojo.players[i].pips
            print("Hand and crib points for this hand were {0} - {1}".format(handScores[0], handScores[1]))
            print("Overall score for this hand was " + self.cribbageDojo.scoreString())
            self.cribbageDojo.resetGame()

            cumulativePegging = cumulativePegging + pegScores[0] - pegScores[1]
            cumulativeHands = cumulativeHands + handScores[0] - handScores[1]
            # Assign the opposite hands to the players
            for i in range(len(hands[0])):
                self.cribbageDojo.players[1].hand.append(Card(hands[0][i].rank, hands[0][i].suit))
                self.cribbageDojo.players[0].hand.append(Card(hands[1][i].rank, hands[1][i].suit))

            for i in range(self.numPlayers):
                pegScores[i] = 0
                handScores[i] = 0

            # print(self.cribbageDojo.players[0].getName()+" has the cards "+cardsString(self.cribbageDojo.players[0].playhand))
            # print(self.cribbageDojo.players[1].getName()+" has the cards "+cardsString(self.cribbageDojo.players[1].playhand))


            self.cribbageDojo.dealer = 1
            self.cribbageDojo.createCrib()
            self.cribbageDojo.cut(starterCard)
            #print("The starter card is cut: "+str(self.cribbageDojo.starter))
            self.cribbageDojo.play()
            for i in range(self.numPlayers):
                pegScores[i] = self.cribbageDojo.players[i].pips
            print("Pegging for this hand was {0} - {1}".format(pegScores[0], pegScores[1]))
            self.cribbageDojo.scoreHands()
            for i in range(self.numPlayers):
                handScores[i] = self.cribbageDojo.players[i].pips - pegScores[i]
                scores[i] = scores[i] + self.cribbageDojo.players[i].pips
            print("Hand and crib points for this hand were {0} - {1}".format(handScores[0], handScores[1]))
            print("Overall score for this hand was " + self.cribbageDojo.scoreString())
            self.cribbageDojo.resetGame()

            cumulativePegging = cumulativePegging + pegScores[0] - pegScores[1]
            cumulativeHands = cumulativeHands + handScores[0] - handScores[1]
            
            print("The point differential for "+self.cribbageDojo.players[0].getName()+" was {0}.\n".format(scores[0]-scores[1]))
            cumulativeScore = cumulativeScore + scores[0] - scores[1]
           
        print("\nThe pegging differential for "+self.cribbageDojo.players[0].getName()+" was {0}, or {1} points-per-hand.\n".format(cumulativePegging,cumulativePegging/(numTrials*2)))
        print("The hand scores differential for "+self.cribbageDojo.players[0].getName()+" was {0}, or {1} points-per-hand.\n".format(cumulativeHands,cumulativeHands/(numTrials*2)))
        print("The overall point differential for "+self.cribbageDojo.players[0].getName()+" was {0}, or {1} points-per-hand.\n".format(cumulativeScore,cumulativeScore/(numTrials*2)))


if __name__ == '__main__':
    players = []
    playersSet = 0

    while playersSet < 2:
        playerName = input("Choose a player (Random, Myrmidon, LinearB, NonLinearB, DeepPeg, Carlo McMonty, Carlos Monteros): ")
        if playerName == "Random":
            playersSet = playersSet + 1
            players.append(PlayerRandom(playersSet,False))
        elif playerName == "Myrmidon":
            playersSet = playersSet + 1
            numSims = int(input("You chose Myrmidon. What number of simulations?: "))
            players.append(Myrmidon(playersSet,numSims,False))
        elif playerName == "LinearB":
            playersSet = playersSet + 1
            alpha = np.max(np.array([0.0,np.min(np.array([1.0,float(input("You chose LinearB. What step size?: "))]))]))
            Lambda = np.max(np.array([0.0,np.min(np.array([1.0,float(input("                   What trace decay rate?: "))]))]))
            players.append(LinearB(playersSet,alpha,Lambda,False))
        elif playerName == "NonLinearB":
            playersSet = playersSet + 1
            alpha = np.max(np.array([0.0,np.min(np.array([1.0,float(input("You chose NonLinearB. What step size?: "))]))]))
            Lambda = np.max(np.array([0.0,np.min(np.array([1.0,float(input("                      What trace decay rate?: "))]))]))
            players.append(NonLinearB(playersSet,alpha,Lambda,False))
        elif playerName == "DeepPeg":
            playersSet = playersSet + 1
            players.append(DeepPeg(playersSet,True,False))
        elif playerName == "Carlo McMonty":
            playersSet = playersSet + 1
            players.append(Monty(playersSet,False))
        elif playerName == "Carlos Monteros":
            playersSet = playersSet + 1
            players.append(Monty2(playersSet,False))
        elif playerName == "exit":
            raise SystemExit(0)
        else:
            print("You chose \""+playerName+"\", which is not a valid option. Please choose again.")

    numTrials = int(input("How many hands do you want to play?: "))

    trainer = TrainHand(players)
    trainer.runTrials(numTrials)
