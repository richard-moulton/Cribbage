#!/usr/bin/env python3

################################################################################
#
# File : Utilities.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : Useful functions that are used throughout the project.
#
# Notes :
#
# Dependencies:
#    - Deck.py (in local project)
#    - numpy (standard python library)
#    - random (standard python library)
#
################################################################################

# Cribbage imports
from Deck import *

# Utility imports
import numpy as np
import random

# Formats a list of cards (usually a hand) as a string
def cardsString(hand):
    if len(hand) > 0:
        cstring = "("
        for card in hand:
            cstring += str(card) + ", "
        return cstring[:-2] + ")"
    else:
        return "(EMPTY)"     

# Checks if two groups of cards (usually cards thrown into the crib) are the same
def areCardsEqual(group1,group2):
    if len(group1)==len(group2):
        group1.sort()
        group2.sort()
        
        for i in range(0,len(group1)):
            if not(group1[i].isIdentical(group2[i])):
                return False
            
        return True
    else:
        return False
    

# Calculates the policy for an array of values using a soft-max selection rule
# see, for example, Sutton and Barto p. 37
def softmax(x):
    s = np.exp(x) / np.sum(np.exp(x))
    return s

# Calculates the policy for an array of values using an epsilon soft rule
# see, for example, Sutton and Barto p. 101
def epsilonsoft(x, epsilon):
    x = [val - min(x) for val in x]
    inds = [i for i, v in enumerate(x) if v == max(x)]
    ind = random.choice(inds)
    den = len(x)
    for i in range(len(x)):
        if i == ind:
            x[i] = 1 - epsilon + (epsilon / den)
        else:
            x[i] = epsilon / den

    return x