#!/usr/bin/env python3

################################################################################
#
# File : trainingscript.py
# Authors : AJ Marasco and Richard Moulton
#
# Description : Automates the processes of training agents against each other
#               and of running round robin tournaments.
#
# Notes :
#
# Dependencies:
#    - Arena.py (in local project)
#    - PlayerRandom.py (in local project)
#    - Myrmidon.py (in local project)
#    - LinearB.py (in local project)
#    - NonLinearB.py (in local project)
#    - DeepPeg.py (in local project)
#    - Monty.py (in local project)
#    - Monty2.py (in local project)
#    - numpy (standard python library)
#
################################################################################

# Cribbage imports
from Arena import Arena

# PLayer imports
from PlayerRandom import PlayerRandom
from Myrmidon import Myrmidon
from LinearB import LinearB
from NonLinearB import NonLinearB
from DeepPeg import DeepPeg
from Monty import Monty
from Monty2 import Monty2

# Utility imports
import numpy as np

# Variables
trainFlag = False
tournamentFlag = True
learningAgents = [LinearB(1,0.5,0.9,False),NonLinearB(1,0.3,0.7,False),DeepPeg(1,True,False),Monty(1,False),Monty2(1,False)]
opponentAgents = [LinearB(2,0,0,False),NonLinearB(2,0,0,False),DeepPeg(2,False,False),Monty(2,False),Monty2(2,False),PlayerRandom(2,False),Myrmidon(2,5,False)]

# Training
if trainFlag:
    for i in range(20):
        for j in range(len(learningAgents)):
            for k in range(len(opponentAgents)):
                if j != k:
                    player1 = learningAgents[j]
                    player2 = opponentAgents[k]
                    arena = Arena([player1,player2],False)
                    arena.playHands(10)
            

# Tournament
if tournamentFlag:
    numAgents = len(opponentAgents)
    peggingResults = np.zeros((numAgents,numAgents))
    handResults = np.zeros((numAgents,numAgents))
    totalResults = np.zeros((numAgents,numAgents))
    for i in range(numAgents):
        for j in range(numAgents):
            if i != j:
                player1 = opponentAgents[i]
                player1.number = 1
                player2 = opponentAgents[j]
                arena = Arena([player1,player2],False)
                matchupResults = arena.playHands(1)
                peggingResults[i][j] = np.average(matchupResults[0])
                handResults[i][j] = np.average(matchupResults[1])
                totalResults[i][j] = np.average(matchupResults[2])
    
    for i in range(numAgents):
        for j in range(numAgents):
            if i < j:
                peggingResults[i][j] = np.average([peggingResults[i][j],-1*peggingResults[j][i]])
                handResults[i][j] = np.average([handResults[i][j],-1*handResults[j][i]])
                totalResults[i][j] = np.average([totalResults[i][j],-1*totalResults[j][i]])
            elif i > j:
                peggingResults[i][j] = -1 * peggingResults[j][i]
                handResults[i][j] = -1 * handResults[j][i]
                totalResults[i][j] = -1 * totalResults[j][i]
    
    peggingAverages = (numAgents * np.average(peggingResults,axis=1))/(numAgents-1)
    handAverages = (numAgents * np.average(handResults,axis=1))/(numAgents-1)
    totalAverages = (numAgents * np.average(totalResults,axis=1))/(numAgents-1)

    
    
