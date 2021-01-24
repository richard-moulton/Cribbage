#!/usr/bin/env python3

################################################################################
#
# File : Cribbage.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : Main file for playing a game of cribbage. Plays through hands
#               one at a time, scoring for nobs, pegging and hands as it goes.
#               Winner is declared as soon as one player reaches 121 points.
#
# Notes : Although this class is mostly written flexibly to allow for three or
#         four player cribbage, it has not been tested for these games. As well,
#         the game does not have a conception of players being on the same team.
#
#         The verboseFlag is used to control whether or not the print commands
#         are used throughout the file. 
#
# Dependencies:
#    - Deck.py (in local project)
#    - Scoring.py (in local project)
#    - Utilities.py (in local project)
#    - random (standard python library)
#
################################################################################

# Cribbage imports
from Deck import Rank,Deck,RiggedDeck
from Scoring import getScore,scoreCards

# Utility imports
from Utilities import cardsString,areCardsEqual
import random

class Cribbage:
    def __init__(self, playerArray, critic = None, verboseFlag = True, rigged=False):
        # Build a single standard deck
        self.rigged = rigged
        self.createDeck()
        # Initialize and empty crib
        self.crib = []
        # Initialize a holder for the cut card
        self.starter = []
        # Initialize the list of played cards
        self.playorder = []
        # Initialize the list of cards currently counting
        self.inplay = []
        # Randomly select which player starts with the crib
        # also the dealer
        self.dealer = random.randint(0, len(playerArray) - 1)
        # initialize the players
        self.players = playerArray
        self.critic = critic

        # determine how much printing should occur
        self.verbose = verboseFlag

    # Reset the game's state, but keep the same players. For use during extended
    # training sessions between players.
    def resetGame(self):
        self.createDeck()
        self.deck.shuffle()
        self.crib = []
        self.starter = []
        self.playorder = []
        self.dealer = random.choice(range(len(self.players)))
        for player in self.players:
            player.newGame(self.gameState())

    # Create and return a dictionary structure capturing the game's state. For
    # use by agents to learn or make decisions.
    def gameState(self):
        state = dict()
        scores = [player.pips for player in self.players]
        state['scores'] = scores
        numCards = [len(player.playhand) for player in self.players]
        state['numCards'] = numCards
        state['inplay'] = self.inplay
        state['playorder'] = self.playorder
        state['dealer'] = self.dealer
        state['starter'] = self.starter
        state['count'] = sum([card.value() for card in self.inplay])

        return state

    # Check to see if any player has won
    def checkWin(self):
        for player in self.players:
            if player.pips > 120:
                return player.number
        else:
            return 0

    # Play a single hand of cribbage
    def playHand(self):
        self.deal()
        self.createCrib()
        self.cut()
        if self.verbose:
            self.show()
        self.play()
        
        # self.scoreHands()
        if self.verbose:
            print("Score is " + self.scoreString())
            print("*******************************")
        self.restoreDeck()
        for player in self.players:
            player.reset(self.gameState())

    # Play a complete game of cribbage - first to 121 wins!
    def playGame(self):
        while not (self.checkWin()):
            self.playHand()

        # print("{} wins! The final score was ".format(self.players[self.checkWin() - 1].getName()) + self.scoreString())
        return self.players[0].pips - self.players[1].pips

    # Deal the initial hands to each player
    def deal(self):
        # Shuffle the deck
        self.deck.shuffle()
        # deal 6 cards to each player, starting to the left of the dealer
        dealOrder = [x % len(self.players) for x in range(self.dealer + 1, self.dealer + len(self.players) + 1)]
        for i in range(6):
            for player in dealOrder:
                self.players[player].draw(self.deck)

    # Each player throws cards into the crib. For 2-player cribbage, each
    # player throws 2 cards into the crib.
    def createCrib(self):
        for player in self.players:
            thrown = player.throwCribCards(2, self.gameState())
            if not(self.critic is None) and player.number == 1:
                criticThrows = self.critic.throwCribCards(2,self.gameState())
                if not(areCardsEqual(criticThrows,thrown)):
                    self.players[0].explainThrow()
                    self.critic.explainThrow()
            if self.verbose:
                print("{} threw 2 cards into the crib.".format(player.getName()))
            for card in thrown:
                self.crib.append(card)

    # Cut the deck to determine the starter card that will be added to all hands
    # If a card is passed as an argument then it is used as the cut card.
    def cut(self, card=None):
        if card is None:
            # Cut the deck
            self.deck.cut()
            # Top card is the starter
            self.starter = self.deck.cards.pop()
        else:
            self.starter = card
        # If starter is a jack, dealer gets 2 pips
        if self.starter.rank is Rank.Jack:
            self.players[self.dealer].pips += 2
            if self.verbose:
                print("{} scores 2 for nobs!".format(self.players[self.dealer].getName()))

    # Score hands in the proper order    
    def scoreHands(self):
        for i in range(self.dealer + 1, self.dealer + 1 + len(self.players)):
            if self.checkWin():
                break
            player = self.players[i % len(self.players)]
            score = getScore(player.hand, self.starter, self.verbose)
            player.pips += score
            if self.verbose:
                print("Scoring {}'s hand: ".format(player.getName()) + cardsString(player.hand) + " + " + str(
                    self.starter))
                print("\t{}'s hand scored {}".format(player.getName(), score))

        if not (self.checkWin()):
            cribScore = getScore(self.crib, self.starter, self.verbose)
            self.players[self.dealer].pips += cribScore
            if self.verbose:
                print(
                    "In {}'s crib: ".format(self.players[self.dealer].getName()) + cardsString(self.crib) + " + " + str(
                        self.starter))
                print("{} scored {} in the crib!\n\n".format(self.players[self.dealer].getName(), cribScore))
        
        for player in self.players:
            player.learnFromHandScores([getScore(self.players[0].hand, self.starter, False), getScore(self.players[1].hand, self.starter, False), getScore(self.crib, self.starter, False)], self.gameState())

    # Play the pegging phase of the game
    def play(self):
        if self.verbose:
            print("{} dealt this hand.".format(self.players[self.dealer].getName()))
        
        if not(self.critic is None):
            self.critic.playhand = []
            for i in range(0,4):
                self.critic.playhand.append(self.players[0].playhand[i])
            #self.players[0].show()
            #self.critic.show()
        
        self.playorder = []
        
        # Starting player is not the dealer
        toPlay = (self.dealer + 1) % len(self.players)
        self.playorder = []
        # as long as any player has cards in hand, and the game isn't over
        while (any(len(player.playhand) > 0 for player in self.players)) and (not (self.checkWin())):
            self.inplay = []  # those cards that affect the current count
            count = 0  # the current count
            goCounter = 0  # a counter for the number of consecutive "go"s

            while (count < 31) and (goCounter < 2) and (not (self.checkWin())):
                if self.verbose:
                    print("It is {}'s turn. Score is ".format(self.players[toPlay].getName()) + self.scoreString())
                # Call on agent to choose a card
                if toPlay == 0 and not(self.critic is None):
                    criticCard = self.critic.playCard(self.gameState())
                    if not(criticCard is None):
                        self.critic.playhand.append(criticCard)
                else:
                    criticCard = None
                playedCard = self.players[toPlay].playCard(self.gameState())
                if playedCard is None:
                    if goCounter == 0:
                        goCounter = 1
                    else:
                        goCounter = 2
                        self.players[toPlay].pips += 1
                        if self.verbose:
                            print("{} scores 1 for the go.\n".format(self.players[toPlay].getName()))
                else:
                    if not(criticCard is None):
                        if not(criticCard.isIdentical(playedCard)):
                            self.players[0].explainPlay()
                            self.critic.explainPlay()
                        else:
                            print("{} agrees with {}'s play.".format(self.critic.getName(),self.players[0].getName()))
                        self.critic.removeCard(playedCard)
                    count += playedCard.value()
                    self.inplay.append(playedCard)
                    self.playorder.append(playedCard)
                    if self.verbose:
                        print("\t{}: ".format(count) + cardsString(self.inplay))
                    self.players[toPlay].pips += scoreCards(self.inplay, self.verbose)
                    goCounter = 0

                toPlay = ((toPlay + 1) % len(self.players))
                # Allow agent to learn from the previous round of plays
                self.players[toPlay].learnFromPegging(self.gameState())

            if goCounter == 2:
                # A go has happened
                for player in self.players:
                    player.go(self.gameState())

            if count == 31:
                pass
                # A 31 has happened
                for player in self.players:
                    player.thirtyOne(self.gameState())

            if self.checkWin():
                # Someone won
                for player in self.players:
                    player.endOfGame(self.gameState())
                if self.verbose:
                    print('Game Over!')

    # Restore the deck after a hand and "pass it" to the next dealer
    def restoreDeck(self):
        self.dealer = ((self.dealer + 1) % len(self.players))
        self.createDeck()
        self.crib = []
        self.starter = []
        self.playorder = []

    # Initalizes own deck depending on whether or not a rigged deck should be used
    def createDeck(self):
        if self.rigged:
            self.deck = RiggedDeck(1)
        else:
            self.deck = Deck(1)

    # Utility functions
    def scoreString(self):
        return str(self.players[0].pips) + " - " + str(self.players[1].pips)

    # Prints out the cards held by each player, the starter card and the cards
    # that have been thrown into the crib
    def show(self):
        for player in self.players:
            print(player)
            
        print("Cut: " + str(self.starter))
        print("Crib: " + cardsString(self.crib))