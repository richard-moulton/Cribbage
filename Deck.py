#!/usr/bin/env python3

################################################################################
#
# File : Deck.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : Classes representing Suits, Ranks, Cards and Decks.
#
# Notes : Every card has a suit and rank. A card's value is what it contributes
#         to the count according to the rules of cribbage.
#
# Dependencies:
#    - enum (standard python library)
#    - random (standard python library)
#
################################################################################

from enum import Enum
import random

class Suit(Enum):
    Spades = 1
    Hearts = 2
    Diamonds = 3
    Clubs = 4

class Rank(Enum):
    Ace = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9
    Ten = 10
    Jack = 11
    Queen = 12
    King = 13

class Card:
    def __init__(self, rank, suit):
        if type(suit) == int:
            self.suit = Suit(suit)
        else:
            self.suit = suit

        if type(rank) == int:
            self.rank = Rank(rank)
        else:
            self.rank = rank

    def __lt__(self, other):
        return self.rank.value < other.rank.value

    def __gt__(self, other):
        return self.rank.value > other.rank.value

    def __eq__(self, other):
        return self.rank.value == other.rank.value

    def uid(self):
        return 13 * (self.suit.value - 1) + self.rank.value

    def getRank(self):
        return self.rank

    def getSuit(self):
        return self.suit

    def value(self):
        if self.rank.value > 10:
            return 10
        else:
            return self.rank.value

    def __str__(self):
        symbols = ["", u"\u2660", u"\u2661", u"\u2662", u"\u2663"]
        vals = ["0", "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        return "{} {}".format(symbols[self.suit.value], vals[self.rank.value])

    def show(self):
        print(str(self))

    def explain(self):
        print("{} of {} has a value of {} and a uid of {}".format(self.rank.name, self.suit.name, self.value(),
                                                                  self.uid()))

    def isIdentical(self, card):
        return card.uid() == self.uid()

class Deck:
    def __init__(self, numDecks):
        self.cards = []
        for i in range(1, numDecks + 1):
            self.build()

    def build(self):
        for suit in Suit:
            for rank in Rank:
                self.cards.append(Card(rank, suit))

    def shuffle(self):
        for i in range(random.randint(3, 10)):
            random.shuffle(self.cards)

    def cut(self):
        t = random.randint(1, len(self.cards))
        self.cards = self.cards[t:] + self.cards[:t]

    def deal(self, players, numCards):
        for i in range(1, numCards + 1):
            for player in players:
                player.draw(self)

    def show(self):
        for card in self.cards:
            card.show()

class RiggedDeck(Deck):
    def __init__(self, numDecks):
        super().__init__(numDecks)
        self.RNG = random.Random(1)

    def shuffle(self):
        self.RNG.seed(1)
        for i in range(2):
            self.RNG.shuffle(self.cards)

    def cut(self):
        self.RNG.seed(2)
        t = self.RNG.randint(1, len(self.cards))
        self.cards = self.cards[t:] + self.cards[:t]