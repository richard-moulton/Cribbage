#!/usr/bin/env python3

################################################################################
#
# File : Scoring.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : Scores cards according to the rules of cribbage.
#
# Notes : There are essentially two types of functions: those that score entire
#         hands and those that score the count during pegging.
#
#         The verboseFlag is used throughout to control whether or not print
#         commands are used.
#
# Dependencies:
#    - Deck.py (in local project)
#    - Utilities.py (in local project)
#    - itertools (standard python library)
#    - math (standard python library)
#
################################################################################

from Deck import *
from Utilities import *
from itertools import combinations
from math import factorial

# These functions score a given hand and starter card.
def getScore(hand, starter, verbose):
    pips = 0
    # Check scoring where the starter card matters
    pips += checkNobs(hand, starter, verbose)
    pips += checkFlush(hand, starter, verbose)
    # Check scoring where the starter card is irrelevant
    hand = hand + [starter]
    pips += checkPairs(hand, verbose)
    for numCards in range(2, len(hand) + 1):
        for combination in combinations(hand, numCards):
            pips += checkSum(combination, 15, verbose)
    pips += checkRuns(hand, verbose)
    return pips

def checkPairs(hand, verbose):
    pips = 0
    for i in range(0, len(hand)):
        for j in range(i + 1, len(hand)):
            if hand[i].rank == hand[j].rank:
                pips += 2
                if verbose:
                    print("\tPair for 2! " + cardsString([hand[i], hand[j]]))

    return pips

def checkNobs(hand, starter, verbose):
    for card in hand:
        if card.rank == Rank.Jack and card.suit == starter.suit:
            if verbose:
                print("\t1 for Nobs! " + cardsString([card, starter]))
            return 1
    return 0

def checkSum(cards, goal, verbose):
    pips = 0
    if (sum([card.value() for card in cards])) == goal:
        if verbose:
            print("\t" + str(goal) + " for 2! " + cardsString(cards))
        pips += 2
    return pips

def checkRuns(hand, verbose):
    pips = 0
    hand.sort(key=lambda card: card.rank.value)
    # check for runs starting with 5
    for i in range(5, 2, -1):
        runFound = False
        for combination in combinations(hand, i):
            if all([x.rank.value - y.rank.value == 1 for x, y in zip(combination[1:], combination[:-1])]):
                if verbose:
                    print("\tRun for " + str(i) + "! " + cardsString(combination))
                pips += i
                runFound = True

        if runFound:
            return pips

    return pips

def checkFlush(hand, starter, verbose):
    pips = 0
    suits = [card.suit == hand[0].suit for card in hand]
    if all(suits):
        if starter.suit == hand[0].suit:
            pips = 5
            temp = "Flush of 5! " + cardsString([starter] + hand)
        else:
            pips = 4
            temp = "Flush of 4! " + cardsString(hand)
        if verbose:
            print("\t" + temp)

    return pips


# These functions score the count during pegging.
def scoreCards(countCards, verbose):
    pips = 0
    run = 0
    pairs = 0

    # Check for a 15 or 31
    pips += checkSum(countCards, 15, verbose)
    pips += checkSum(countCards, 31, verbose)

    # Check for runs
    for i in range(3, len(countCards) + 1):
        run = max(run, scoreRun(countCards[len(countCards) - i:len(countCards)]))

    # check for pairs
    for i in range(2, min(len(countCards), 4)+1):
        pairs = max(pairs, scorePairs(countCards[-i:]))

    if (run > 0) and verbose:
        print("That's a run of {}!".format(run))

    if (pairs > 0) and verbose:
        print("That's {} pair!".format(int(pairs / 2)))

    pips += run
    pips += pairs

    return pips

def scoreRun(cards):
    pips = 0
    cards.sort(key=lambda card: card.rank.value)
    if all([x.rank.value - y.rank.value == 1 for x, y in zip(cards[1:], cards[:-1])]):
        pips = len(cards)

    return pips

def scorePairs(cards):
    pips = 0
    if all([x.rank.value - y.rank.value == 0 for x, y in zip(cards[1:], cards[:-1])]):
        pips = len(cards) * (len(cards) - 1)

    return pips