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
from CriticSessions import CriticSessions

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
tournamentFlag = False
criticFlag = True

verboseFlag = True

# Training
if trainFlag:
    learningAgents = [DeepPeg(1,False,True,False)]#[LinearB(1,0.5,0.9,False),NonLinearB(1,0.3,0.7,False),DeepPeg(1,True,False),Monty(1,False),Monty2(1,False)]
    opponentAgents = [Myrmidon(2,5,False),DeepPeg(2,False,False,False)]

    for i in range(50):
        for j in range(len(learningAgents)):
            for k in range(len(opponentAgents)):
                player1 = learningAgents[j]
                player2 = opponentAgents[k]
                arena = Arena([player1,player2],False,verboseFlag)
                arena.playHands(10)
            

# Tournament
if tournamentFlag:
    opponentAgents = [PlayerRandom(1,False),DeepPeg(1,True,False,False),DeepPeg(1,False,False,False),Myrmidon(2,1,False),Myrmidon(2,5,False),Myrmidon(2,10,False)]
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
                player2.number = 2
                arena = Arena([player1,player2],False,verboseFlag)
                matchupResults = arena.playHands(50)
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

# Critic
if criticFlag:
        player1 = DeepPeg(1,False,True,False)
        player2 = Myrmidon(2,5,False)
        critic = Myrmidon(0,5,False)
        criticSession = CriticSessions([player1,player2],critic,verboseFlag)
        criticSession.playHands(2)
    
