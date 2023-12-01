import math
from random import randrange
import copy
import DQNPacmanAgentV2 as dpa
import DQNPacmanParametersV2 as dp
import numpy as np
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque,namedtuple
import random
import pickle
import itertools
import time
import smtplib
from datetime import datetime
import os
from email.message import EmailMessage
from email.utils import formataddr
import matplotlib.pyplot as plt



# 28 Across 31 Tall 1: Empty Space 2: Tic-Tak 3: Wall 4: Ghost safe-space 5: Special Tic-Tak
originalGameBoard = [
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,6,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,6,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,2,3],
    [3,2,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,2,2,3],
    [3,3,3,3,3,3,2,3,3,3,3,3,1,3,3,1,3,3,3,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,3,3,3,1,3,3,1,3,3,3,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,1,1,1,1,1,1,1,1,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,4,4,4,4,4,4,3,1,3,3,2,3,3,3,3,3,3],
    [1,1,1,1,1,1,2,1,1,1,3,4,4,4,4,4,4,3,1,1,1,2,1,1,1,1,1,1], # Middle Lane Row: 14
    [3,3,3,3,3,3,2,3,3,1,3,4,4,4,4,4,4,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,1,1,1,1,1,1,1,1,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,6,2,2,3,3,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,3,3,2,2,6,3],
    [3,3,3,2,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,2,3,3,3],
    [3,3,3,2,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,2,3,3,3],
    [3,2,2,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3],
    [3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
]

####### PACMAN GAME IMPLEMENTATION, NOT MY CODE ######## - But was adapted for my use!

gameBoard = copy.deepcopy(originalGameBoard)
# spriteRatio = 3/2
# square = 25 # Size of each unit square
# spriteOffset = square * (1 - spriteRatio) * (1/2)
# (width, height) = (len(gameBoard[0]) * square, len(gameBoard) * square) # Game screen
# screen = pygame.display.set_mode((width, height))
# pygame.display.flip()
# musicPlaying = 0 # 0: Chomp, 1: Important, 2: Siren
# # pelletColor = (165, 93, 53)
# pelletColor = (222, 161, 133)

# PLAYING_KEYS = {
#     "up":[pygame.K_w, pygame.K_UP],
#     "down":[pygame.K_s, pygame.K_DOWN],
#     "right":[pygame.K_d, pygame.K_RIGHT],
#     "left":[pygame.K_a, pygame.K_LEFT]
# }

class Game:
    def __init__(self, level, score):
        self.paused = True
        self.ghostUpdateDelay = 1
        self.ghostUpdateCount = 0
        self.pacmanUpdateDelay = 1
        self.pacmanUpdateCount = 0
        self.tictakChangeDelay = 10
        self.tictakChangeCount = 0
        self.ghostsAttacked = False
        self.highScore = self.getHighScore()
        self.score = score
        self.level = level
        self.lives = 1
        self.ghosts = [Ghost(14, 13, "red", 0), Ghost(17, 11, "blue", 1), Ghost(17, 13, "pink", 2), Ghost(17, 15, "orange", 3)]
        self.pacman = Pacman(26, 13) # Center of Second Last Row
        self.total = self.getCount()
        self.ghostScore = 200
        self.levels = [[350, 250], [150, 450], [150, 450], [0, 600]]

        for i in range(self.level):
            for lvl in self.levels:
                lvl[0] = min((lvl[0] + lvl[1]) - 100, lvl[0] + 50)
                lvl[1] = max(100, lvl[1] - 50)

        random.shuffle(self.levels)
        # Level index and Level Progress
        self.ghostStates = [[1, 0], [0, 0], [1, 0], [0, 0]]
        index = 0
        for state in self.ghostStates:
            state[0] = randrange(2)
            state[1] = randrange(self.levels[index][state[0]] + 1)
            index += 1
        self.collected = 0
        self.started = False 
        self.gameOver = False
        self.gameOverCounter = 0
        self.points = []
        self.pointsTimer = 10
        # Berry Spawn Time, Berry Death Time, Berry Eaten
        self.berryState = [200, 400, False]
        self.berryLocation = [20, 13]
        self.berries = ["tile080.png", "tile081.png", "tile082.png", "tile083.png", "tile084.png", "tile085.png", "tile086.png", "tile087.png"]
        self.berriesCollected = []
        self.levelTimer = 0
        self.berryScore = 100
        self.lockedInTimer = 100
        self.lockedIn = True
        self.extraLifeGiven = False
        self.musicPlaying = 0
        self.killed_by_ghosts = False


    # Driver method: The games primary update method
    def update(self):
        # pygame.image.unload()
        #print(self.ghostStates)
        if self.gameOver:
            self.gameOverFunc()
            
            return
        # if self.paused or not self.started:
        #     self.drawTilesAround(21, 10)
        #     self.drawTilesAround(21, 11)
        #     self.drawTilesAround(21, 12)
        #     self.drawTilesAround(21, 13)
        #     self.drawTilesAround(21, 14)
        #     self.drawReady()
        #     pygame.display.update()
        #     return

        self.levelTimer += 1
        self.ghostUpdateCount += 1
        self.pacmanUpdateCount += 1
        self.tictakChangeCount += 1
        self.ghostsAttacked = False

        # if self.score >= 10000 and not self.extraLifeGiven:
        #     self.lives += 1
        #     self.extraLifeGiven = True
        #     self.forcePlayMusic("pacman_extrapac.wav")

        # Draw tiles around ghosts and pacman
        # self.clearBoard()
        for ghost in self.ghosts:
            if ghost.attacked:
                self.ghostsAttacked = True

        # Check if the ghost should chase pacman
        index = 0
        for state in self.ghostStates:
            state[1] += 1
            if state[1] >= self.levels[index][state[0]]:
                state[1] = 0
                state[0] += 1
                state[0] %= 2
            index += 1

        index = 0
        for ghost in self.ghosts:
            if not ghost.attacked and not ghost.dead and self.ghostStates[index][0] == 0:
                ghost.target = [self.pacman.row, self.pacman.col]
            index += 1

        if self.levelTimer == self.lockedInTimer:
            self.lockedIn = False

        self.checkSurroundings()
        if self.ghostUpdateCount == self.ghostUpdateDelay:
            for ghost in self.ghosts:
                ghost.update()
            self.ghostUpdateCount = 0

        if self.tictakChangeCount == self.tictakChangeDelay:
            #Changes the color of special Tic-Taks
            #self.flipColor()
            self.tictakChangeCount = 0

        if self.pacmanUpdateCount == self.pacmanUpdateDelay:
            self.pacmanUpdateCount = 0
            self.checkSurroundings()
            self.pacman.update()
            self.pacman.col %= len(gameBoard[0])
            if self.pacman.row % 1 == 0 and self.pacman.col % 1 == 0:
                if gameBoard[int(self.pacman.row)][int(self.pacman.col)] == 2:
                    # self.playMusic("munch_1.wav")
                    gameBoard[int(self.pacman.row)][int(self.pacman.col)] = 1
                    self.score += 10
                    self.collected += 1
                    # # Fill tile with black
                    # pygame.draw.rect(screen, (0, 0, 0), (self.pacman.col * square, self.pacman.row * square, square, square))
                elif gameBoard[int(self.pacman.row)][int(self.pacman.col)] == 5 or gameBoard[int(self.pacman.row)][int(self.pacman.col)] == 6:
                    # self.forcePlayMusic("power_pellet.wav")
                    gameBoard[int(self.pacman.row)][int(self.pacman.col)] = 1
                    self.collected += 1
                    # Fill tile with black
                    # pygame.draw.rect(screen, (0, 0, 0), (self.pacman.col * square, self.pacman.row * square, square, square))
                    self.score += 50
                    self.ghostScore = 200
                    for ghost in self.ghosts:
                        ghost.attackedCount = 0
                        ghost.setAttacked(True)
                        ghost.setTarget()
                        self.ghostsAttacked = True
        self.checkSurroundings()
        self.highScore = max(self.score, self.highScore)
        #print(gameBoard)

        global running
        if self.collected == self.total:
            # print("New Level")
            # self.forcePlayMusic("intermission.wav")
            # self.level += 1
            # self.newLevel()
            print("You win", self.level)
            self.gameOver = True
            self.gameOverFunc()

        # if self.level - 1 == 8: #(self.levels[0][0] + self.levels[0][1]) // 50:
        #     print("You win", self.level, len(self.levels))
        #     running = False
        # self.softRender()

    # Render method
    # def render(self):
    #     screen.fill((0, 0, 0)) # Flushes the screen
    #     # Draws game elements
    #     currentTile = 0
    #     self.displayLives()
    #     self.displayScore()
    #     for i in range(3, len(gameBoard) - 2):
    #         for j in range(len(gameBoard[0])):
    #             if gameBoard[i][j] == 3: # Draw wall
    #                 imageName = str(currentTile)
    #                 if len(imageName) == 1:
    #                     imageName = "00" + imageName
    #                 elif len(imageName) == 2:
    #                      imageName = "0" + imageName
    #                 # Get image of desired tile
    #                 imageName = "tile" + imageName + ".png"
    #                 tileImage = pygame.image.load(BoardPath + imageName)
    #                 tileImage = pygame.transform.scale(tileImage, (square, square))

    #                 #Display image of tile
    #                 screen.blit(tileImage, (j * square, i * square, square, square))

    #                 # pygame.draw.rect(screen, (0, 0, 255),(j * square, i * square, square, square)) # (x, y, width, height)
    #             elif gameBoard[i][j] == 2: # Draw Tic-Tak
    #                 pygame.draw.circle(screen, pelletColor,(j * square + square//2, i * square + square//2), square//4)
    #             elif gameBoard[i][j] == 5: #Black Special Tic-Tak
    #                 pygame.draw.circle(screen, (0, 0, 0),(j * square + square//2, i * square + square//2), square//2)
    #             elif gameBoard[i][j] == 6: #White Special Tic-Tak
    #                 pygame.draw.circle(screen, pelletColor,(j * square + square//2, i * square + square//2), square//2)

    #             currentTile += 1
    #     # Draw Sprites
    #     for ghost in self.ghosts:
    #         ghost.draw()
    #     self.pacman.draw()
    #     # Updates the screen
    #     pygame.display.update()


    # def softRender(self):
    #     pointsToDraw = []
    #     for point in self.points:
    #         if point[3] < self.pointsTimer:
    #             pointsToDraw.append([point[2], point[0], point[1]])
    #             point[3] += 1
    #         else:
    #             self.points.remove(point)
    #             self.drawTilesAround(point[0], point[1])

    #     for point in pointsToDraw:
    #         self.drawPoints(point[0], point[1], point[2])

    #     # Draw Sprites
    #     for ghost in self.ghosts:
    #         ghost.draw()
    #     self.pacman.draw()
    #     self.displayScore()
    #     self.displayBerries()
    #     self.displayLives()
    #     # for point in pointsToDraw:
    #     #     self.drawPoints(point[0], point[1], point[2])
    #     self.drawBerry()
    #     # Updates the screen
    #     pygame.display.update()

    # def playMusic(self, music):
    #     # return False # Uncomment to disable music
    #     global musicPlaying
    #     if not pygame.mixer.music.get_busy():
    #         pygame.mixer.music.unload()
    #         pygame.mixer.music.load(MusicPath + music)
    #         pygame.mixer.music.queue(MusicPath + music)
    #         pygame.mixer.music.play()
    #         if music == "munch_1.wav":
    #             musicPlaying = 0
    #         elif music == "siren_1.wav":
    #             musicPlaying = 2
    #         else:
    #             musicPlaying = 1

    # def forcePlayMusic(self, music):
    #     # return False # Uncomment to disable music
    #     pygame.mixer.music.unload()
    #     pygame.mixer.music.load(MusicPath + music)
    #     pygame.mixer.music.play()
    #     global musicPlaying
    #     musicPlaying = 1

    # def clearBoard(self):
    #         # Draw tiles around ghosts and pacman
    #         for ghost in self.ghosts:
    #             self.drawTilesAround(ghost.row, ghost.col)
    #         self.drawTilesAround(self.pacman.row, self.pacman.col)
    #         self.drawTilesAround(self.berryLocation[0], self.berryLocation[1])
    #         # Clears Ready! Label
    #         self.drawTilesAround(20, 10)
    #         self.drawTilesAround(20, 11)
    #         self.drawTilesAround(20, 12)
    #         self.drawTilesAround(20, 13)
    #         self.drawTilesAround(20, 14)

    def checkSurroundings(self):
        # Check if pacman got killed
        for ghost in self.ghosts:
            if self.touchingPacman(ghost.row, ghost.col) and not ghost.attacked:
                print("You lose")
                self.gameOver = True
                self.killed_by_ghosts = True
                #Removes the ghosts from the screen
                # for ghost in self.ghosts:
                #     self.drawTilesAround(ghost.row, ghost.col)
                # self.drawTilesAround(self.pacman.row, self.pacman.col)
                # self.pacman.draw()
                # pygame.display.update()
                # pause(10000000)
                return
            elif self.touchingPacman(ghost.row, ghost.col) and ghost.isAttacked() and not ghost.isDead():
                ghost.setDead(True)
                ghost.setTarget()
                ghost.ghostSpeed = 1
                ghost.row = math.floor(ghost.row)
                ghost.col = math.floor(ghost.col)
                self.score += self.ghostScore
                self.points.append([ghost.row, ghost.col, self.ghostScore, 0])
                self.ghostScore *= 2
                # self.forcePlayMusic("eat_ghost.wav")
                pause(10000000)
        if self.touchingPacman(self.berryLocation[0], self.berryLocation[1]) and not self.berryState[2] and self.levelTimer in range(self.berryState[0], self.berryState[1]):
            self.berryState[2] = True
            self.score += self.berryScore
            self.points.append([self.berryLocation[0], self.berryLocation[1], self.berryScore, 0])
            self.berriesCollected.append(self.berries[(self.level - 1) % 8])
            # self.forcePlayMusic("eat_fruit.wav")
    # Displays the current score
    # def displayScore(self):
    #     textOneUp = ["tile033.png", "tile021.png", "tile016.png"]
    #     textHighScore = ["tile007.png", "tile008.png", "tile006.png", "tile007.png", "tile015.png", "tile019.png", "tile002.png", "tile014.png", "tile018.png", "tile004.png"]
    #     index = 0
    #     scoreStart = 5
    #     highScoreStart = 11
    #     for i in range(scoreStart, scoreStart+len(textOneUp)):
    #         tileImage = pygame.image.load(TextPath + textOneUp[index])
    #         tileImage = pygame.transform.scale(tileImage, (square, square))
    #         screen.blit(tileImage, (i * square, 4, square, square))
    #         index += 1
    #     score = str(self.score)
    #     if score == "0":
    #         score = "00"
    #     index = 0
    #     for i in range(0, len(score)):
    #         digit = int(score[i])
    #         tileImage = pygame.image.load(TextPath + "tile0" + str(32 + digit) + ".png")
    #         tileImage = pygame.transform.scale(tileImage, (square, square))
    #         screen.blit(tileImage, ((scoreStart + 2 + index) * square, square + 4, square, square))
    #         index += 1

    #     index = 0
    #     for i in range(highScoreStart, highScoreStart+len(textHighScore)):
    #         tileImage = pygame.image.load(TextPath + textHighScore[index])
    #         tileImage = pygame.transform.scale(tileImage, (square, square))
    #         screen.blit(tileImage, (i * square, 4, square, square))
    #         index += 1

    #     highScore = str(self.highScore)
    #     if highScore == "0":
    #         highScore = "00"
    #     index = 0
    #     for i in range(0, len(highScore)):
    #         digit = int(highScore[i])
    #         tileImage = pygame.image.load(TextPath + "tile0" + str(32 + digit) + ".png")
    #         tileImage = pygame.transform.scale(tileImage, (square, square))
    #         screen.blit(tileImage, ((highScoreStart + 6 + index) * square, square + 4, square, square))
    #         index += 1

    # def drawBerry(self):
    #     if self.levelTimer in range(self.berryState[0], self.berryState[1]) and not self.berryState[2]:
    #         # print("here")
    #         berryImage = pygame.image.load(ElementPath + self.berries[(self.level - 1) % 8])
    #         berryImage = pygame.transform.scale(berryImage, (int(square * spriteRatio), int(square * spriteRatio)))
    #         screen.blit(berryImage, (self.berryLocation[1] * square, self.berryLocation[0] * square, square, square))


    # def drawPoints(self, points, row, col):
    #     pointStr = str(points)
    #     index = 0
    #     for i in range(len(pointStr)):
    #         digit = int(pointStr[i])
    #         tileImage = pygame.image.load(TextPath + "tile" + str(224 + digit) + ".png")
    #         tileImage = pygame.transform.scale(tileImage, (square//2, square//2))
    #         screen.blit(tileImage, ((col) * square + (square//2 * index), row * square - 20, square//2, square//2))
    #         index += 1

    # def drawReady(self):
    #     ready = ["tile274.png", "tile260.png", "tile256.png", "tile259.png", "tile281.png", "tile283.png"]
    #     for i in range(len(ready)):
    #         letter = pygame.image.load(TextPath + ready[i])
    #         letter = pygame.transform.scale(letter, (int(square), int(square)))
    #         screen.blit(letter, ((11 + i) * square, 20 * square, square, square))

    def gameOverFunc(self):
        global running
        reset()
        #self.recordHighScore()
        self.gameOver = False
        return

        # Resets the screen around pacman
        # self.drawTilesAround(self.pacman.row, self.pacman.col)

        # Draws new image
        # pacmanImage = pygame.image.load(ElementPath + "tile" + str(116 + self.gameOverCounter) + ".png")
        # pacmanImage = pygame.transform.scale(pacmanImage, (int(square * spriteRatio), int(square * spriteRatio)))
        # screen.blit(pacmanImage, (self.pacman.col * square + spriteOffset, self.pacman.row * square + spriteOffset, square, square))
        # pygame.display.update()
 

    # def displayLives(self):
    #     # 33 rows || 28 cols
    #     # Lives[[31, 5], [31, 3], [31, 1]]
    #     livesLoc = [[34, 3], [34, 1]]
    #     for i in range(self.lives - 1):
    #         lifeImage = pygame.image.load(ElementPath + "tile054.png")
    #         lifeImage = pygame.transform.scale(lifeImage, (int(square * spriteRatio), int(square * spriteRatio)))
    #         screen.blit(lifeImage, (livesLoc[i][1] * square, livesLoc[i][0] * square - spriteOffset, square, square))

    # def displayBerries(self):
    #     firstBerrie = [34, 26]
    #     for i in range(len(self.berriesCollected)):
    #         berrieImage = pygame.image.load(ElementPath + self.berriesCollected[i])
    #         berrieImage = pygame.transform.scale(berrieImage, (int(square * spriteRatio), int(square * spriteRatio)))
    #         screen.blit(berrieImage, ((firstBerrie[1] - (2*i)) * square, firstBerrie[0] * square + 5, square, square))

    def touchingPacman(self, row, col):
        #print(f"row: {row}, column: {col}, pacman row: {self.pacman.row}, pacman col: {self.pacman.col}")
        if row == self.pacman.row and col == self.pacman.col:
            return True
        else:
            return False
        # if row - 0.5 <= self.pacman.row and row >= self.pacman.row and col == self.pacman.col:
        #     return True
        # elif row + 0.5 >= self.pacman.row and row <= self.pacman.row and col == self.pacman.col:
        #     return True
        # elif row == self.pacman.row and col - 0.5 <= self.pacman.col and col >= self.pacman.col:
        #     return True
        # elif row == self.pacman.row and col + 0.5 >= self.pacman.col and col <= self.pacman.col:
        #     return True
        # elif row == self.pacman.row and col == self.pacman.col:
        #     return True
        # else:
        #     return False

    # def newLevel(self):
    #     reset()
    #     #self.lives += 1
    #     self.collected = 0
    #     self.started = False
    #     self.berryState = [200, 400, False]
    #     self.levelTimer = 0
    #     self.lockedIn = True
    #     for level in self.levels:
    #         level[0] = min((level[0] + level[1]) - 100, level[0] + 50)
    #         level[1] = max(100, level[1] - 50)
    #     random.shuffle(self.levels)
    #     index = 0
    #     for state in self.ghostStates:
    #         state[0] = randrange(2)
    #         state[1] = randrange(self.levels[index][state[0]] + 1)
    #         index += 1
    #     global gameBoard
    #     gameBoard = copy.deepcopy(originalGameBoard)
    #     self.render()

    # def drawTilesAround(self, row, col):
    #     row = math.floor(row)
    #     col = math.floor(col)
    #     for i in range(row-2, row+3):
    #         for j in range(col-2, col+3):
    #             if i >= 3 and i < len(gameBoard) - 2 and j >= 0 and j < len(gameBoard[0]):
    #                 imageName = str(((i - 3) * len(gameBoard[0])) + j)
    #                 if len(imageName) == 1:
    #                     imageName = "00" + imageName
    #                 elif len(imageName) == 2:
    #                      imageName = "0" + imageName
    #                 # Get image of desired tile
    #                 imageName = "tile" + imageName + ".png"
    #                 tileImage = pygame.image.load(BoardPath + imageName)
    #                 tileImage = pygame.transform.scale(tileImage, (square, square))
    #                 #Display image of tile
    #                 screen.blit(tileImage, (j * square, i * square, square, square))

    #                 if gameBoard[i][j] == 2: # Draw Tic-Tak
    #                     pygame.draw.circle(screen, pelletColor,(j * square + square//2, i * square + square//2), square//4)
    #                 elif gameBoard[i][j] == 5: #Black Special Tic-Tak
    #                     pygame.draw.circle(screen, (0, 0, 0),(j * square + square//2, i * square + square//2), square//2)
    #                 elif gameBoard[i][j] == 6: #White Special Tic-Tak
    #                     pygame.draw.circle(screen, pelletColor,(j * square + square//2, i * square + square//2), square//2)

    # Flips Color of Special Tic-Taks
    # def flipColor(self):
    #     global gameBoard
    #     for i in range(3, len(gameBoard) - 2):
    #         for j in range(len(gameBoard[0])):
    #             if gameBoard[i][j] == 5:
    #                 gameBoard[i][j] = 6
    #                 pygame.draw.circle(screen, pelletColor,(j * square + square//2, i * square + square//2), square//2)
    #             elif gameBoard[i][j] == 6:
    #                 gameBoard[i][j] = 5
    #                 pygame.draw.circle(screen, (0, 0, 0),(j * square + square//2, i * square + square//2), square//2)

    def getCount(self):
        total = 0
        for i in range(3, len(gameBoard) - 2):
            for j in range(len(gameBoard[0])):
                if gameBoard[i][j] == 2 or gameBoard[i][j] == 5 or gameBoard[i][j] == 6:
                    total += 1
        return total

    def getHighScore(self):
        # file = open(DataPath + "HighScore.txt", "r")
        # highScore = int(file.read())
        # file.close()
        highScore = 100
        return highScore

    def recordHighScore(self):
        pass
        # file = open(DataPath + "HighScore.txt", "w").close()
        # file = open(DataPath + "HighScore.txt", "w+")
        # file.write(str(self.highScore))
        # file.close()

class Pacman:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.mouthOpen = False
        self.pacSpeed = 1
        self.mouthChangeDelay = 5
        self.mouthChangeCount = 0
        self.dir = 0 # 0: North, 1: East, 2: South, 3: West
        self.newDir = 0

    def update(self):
        if self.newDir == 0:
            if canMove(math.floor(self.row - self.pacSpeed), self.col) and self.col % 1 == 0:
                self.row -= self.pacSpeed
                self.dir = self.newDir
                return
        elif self.newDir == 1:
            if canMove(self.row, math.ceil(self.col + self.pacSpeed)) and self.row % 1 == 0:
                self.col += self.pacSpeed
                self.dir = self.newDir
                return
        elif self.newDir == 2:
            if canMove(math.ceil(self.row + self.pacSpeed), self.col) and self.col % 1== 0:
                self.row += self.pacSpeed
                self.dir = self.newDir
                return
        elif self.newDir == 3:
            if canMove(self.row, math.floor(self.col - self.pacSpeed)) and self.row % 1 == 0:
                self.col -= self.pacSpeed
                self.dir = self.newDir
                return

        if self.dir == 0:
            if canMove(math.floor(self.row - self.pacSpeed), self.col) and self.col % 1 == 0:
                self.row -= self.pacSpeed
        elif self.dir == 1:
            if canMove(self.row, math.ceil(self.col + self.pacSpeed)) and self.row % 1 == 0:
                self.col += self.pacSpeed
        elif self.dir == 2:
            if canMove(math.ceil(self.row + self.pacSpeed), self.col) and self.col % 1 == 0:
                self.row += self.pacSpeed
        elif self.dir == 3:
            if canMove(self.row, math.floor(self.col - self.pacSpeed)) and self.row % 1 == 0:
                self.col -= self.pacSpeed

    # Draws pacman based on his current state
    # def draw(self):
    #     if not game.started:
    #         pacmanImage = pygame.image.load(ElementPath + "tile112.png")
    #         pacmanImage = pygame.transform.scale(pacmanImage, (int(square * spriteRatio), int(square * spriteRatio)))
    #         screen.blit(pacmanImage, (self.col * square + spriteOffset, self.row * square + spriteOffset, square, square))
    #         return

    #     if self.mouthChangeCount == self.mouthChangeDelay:
    #         self.mouthChangeCount = 0
    #         self.mouthOpen = not self.mouthOpen
    #     self.mouthChangeCount += 1
    #     # pacmanImage = pygame.image.load("Sprites/tile049.png")
    #     if self.dir == 0:
    #         if self.mouthOpen:
    #             pacmanImage = pygame.image.load(ElementPath + "tile049.png")
    #         else:
    #             pacmanImage = pygame.image.load(ElementPath + "tile051.png")
    #     elif self.dir == 1:
    #         if self.mouthOpen:
    #             pacmanImage = pygame.image.load(ElementPath + "tile052.png")
    #         else:
    #             pacmanImage = pygame.image.load(ElementPath + "tile054.png")
    #     elif self.dir == 2:
    #         if self.mouthOpen:
    #             pacmanImage = pygame.image.load(ElementPath + "tile053.png")
    #         else:
    #             pacmanImage = pygame.image.load(ElementPath + "tile055.png")
    #     elif self.dir == 3:
    #         if self.mouthOpen:
    #             pacmanImage = pygame.image.load(ElementPath + "tile048.png")
    #         else:
    #             pacmanImage = pygame.image.load(ElementPath + "tile050.png")

    #     pacmanImage = pygame.transform.scale(pacmanImage, (int(square * spriteRatio), int(square * spriteRatio)))
    #     screen.blit(pacmanImage, (self.col * square + spriteOffset, self.row * square + spriteOffset, square, square))

class Ghost:
    def __init__(self, row, col, color, changeFeetCount):
        self.row = row
        self.col = col
        self.attacked = False
        self.color = color
        self.dir = randrange(4)
        self.dead = False
        self.changeFeetCount = changeFeetCount
        self.changeFeetDelay = 5
        self.target = [-1, -1]
        self.ghostSpeed = 1#1/4
        self.lastLoc = [-1, -1]
        self.attackedTimer = 70
        self.attackedCount = 0
        self.deathTimer = 120
        self.deathCount = 0

    def update(self):
        #print(self.row, self.col)
        if self.target == [-1, -1] or (self.row == self.target[0] and self.col == self.target[1]) or gameBoard[int(self.row)][int(self.col)] == 4 or self.dead:
            self.setTarget()
        self.setDir()
        self.move()

        if self.attacked:
            self.attackedCount += 1

        if self.attacked and not self.dead:
            self.ghostSpeed = 1 #1/8

        if self.attackedCount == self.attackedTimer and self.attacked:
            if not self.dead:
                self.ghostSpeed = 1 #1/4
                self.row = math.floor(self.row)
                self.col = math.floor(self.col)

            self.attackedCount = 0
            self.attacked = False
            self.setTarget()

        if self.dead and gameBoard[self.row][self.col] == 4:
            self.deathCount += 1
            self.attacked = False
            if self.deathCount == self.deathTimer:
                self.deathCount = 0
                self.dead = False
                self.ghostSpeed = 1 #1/4

    # def draw(self): # Ghosts states: Alive, Attacked, Dead Attributes: Color, Direction, Location
    #     ghostImage = pygame.image.load(ElementPath + "tile152.png")
    #     currentDir = ((self.dir + 3) % 4) * 2
    #     if self.changeFeetCount == self.changeFeetDelay:
    #         self.changeFeetCount = 0
    #         currentDir += 1
    #     self.changeFeetCount += 1
    #     if self.dead:
    #         tileNum = 152 + currentDir
    #         ghostImage = pygame.image.load(ElementPath + "tile" + str(tileNum) + ".png")
    #     elif self.attacked:
    #         if self.attackedTimer - self.attackedCount < self.attackedTimer//3:
    #             if (self.attackedTimer - self.attackedCount) % 31 < 26:
    #                 ghostImage = pygame.image.load(ElementPath + "tile0" + str(70 + (currentDir - (((self.dir + 3) % 4) * 2))) + ".png")
    #             else:
    #                 ghostImage = pygame.image.load(ElementPath + "tile0" + str(72 + (currentDir - (((self.dir + 3) % 4) * 2))) + ".png")
    #         else:
    #             ghostImage = pygame.image.load(ElementPath + "tile0" + str(72 + (currentDir - (((self.dir + 3) % 4) * 2))) + ".png")
    #     else:
    #         if self.color == "blue":
    #             tileNum = 136 + currentDir
    #             ghostImage = pygame.image.load(ElementPath + "tile" + str(tileNum) + ".png")
    #         elif self.color == "pink":
    #             tileNum = 128 + currentDir
    #             ghostImage = pygame.image.load(ElementPath + "tile" + str(tileNum) + ".png")
    #         elif self.color == "orange":
    #             tileNum = 144 + currentDir
    #             ghostImage = pygame.image.load(ElementPath + "tile" + str(tileNum) + ".png")
    #         elif self.color == "red":
    #             tileNum = 96 + currentDir
    #             if tileNum < 100:
    #                 ghostImage = pygame.image.load(ElementPath + "tile0" + str(tileNum) + ".png")
    #             else:
    #                 ghostImage = pygame.image.load(ElementPath + "tile" + str(tileNum) + ".png")

    #     ghostImage = pygame.transform.scale(ghostImage, (int(square * spriteRatio), int(square * spriteRatio)))
    #     screen.blit(ghostImage, (self.col * square + spriteOffset, self.row * square + spriteOffset, square, square))

    def isValidTwo(self, cRow, cCol, dist, visited):
        if cRow < 3 or cRow >= len(gameBoard) - 5 or cCol < 0 or cCol >= len(gameBoard[0]) or gameBoard[cRow][cCol] == 3:
            return False
        elif visited[cRow][cCol] <= dist:
            return False
        return True

    def isValid(self, cRow, cCol):
        if cCol < 0 or cCol > len(gameBoard[0]) - 1:
            return True
        for ghost in game.ghosts:
            if ghost.color == self.color:
                continue
            if ghost.row == cRow and ghost.col == cCol and not self.dead:
                return False
        if not ghostGate.count([cRow, cCol]) == 0:
            if self.dead and self.row < cRow:
                return True
            elif self.row > cRow and not self.dead and not self.attacked and not game.lockedIn:
                return True
            else:
                return False
        if gameBoard[cRow][cCol] == 3:
            return False
        return True

    def setDir(self): #Very inefficient || can easily refactor
        # BFS search -> Not best route but a route none the less
        dirs = [[0, -self.ghostSpeed, 0],
                [1, 0, self.ghostSpeed],
                [2, self.ghostSpeed, 0],
                [3, 0, -self.ghostSpeed]
        ]
        random.shuffle(dirs)
        best = 10000
        bestDir = -1
        for newDir in dirs:
            if self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]]) < best:
                if not (self.lastLoc[0] == self.row + newDir[1] and self.lastLoc[1] == self.col + newDir[2]):
                    if newDir[0] == 0 and self.col % 1 == 0:
                        if self.isValid(math.floor(self.row + newDir[1]), int(self.col + newDir[2])):
                            bestDir = newDir[0]
                            best = self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
                    elif newDir[0] == 1 and self.row % 1 == 0:
                        if self.isValid(int(self.row + newDir[1]), math.ceil(self.col + newDir[2])):
                            bestDir = newDir[0]
                            best = self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
                    elif newDir[0] == 2 and self.col % 1== 0:
                        if self.isValid(math.ceil(self.row + newDir[1]), int(self.col + newDir[2])):
                            bestDir = newDir[0]
                            best = self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
                    elif newDir[0] == 3 and self.row % 1== 0:
                        if self.isValid(int(self.row + newDir[1]), math.floor(self.col + newDir[2])):
                            bestDir = newDir[0]
                            best = self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
        self.dir = bestDir

    def calcDistance(self, a, b):
        dR = a[0] - b[0]
        dC = a[1] - b[1]
        return math.sqrt((dR * dR) + (dC * dC))

    def setTarget(self):
        if gameBoard[int(self.row)][int(self.col)] == 4 and not self.dead:
            self.target = [ghostGate[0][0] - 1, ghostGate[0][1]+1]
            return
        elif gameBoard[int(self.row)][int(self.col)] == 4 and self.dead:
            self.target = [self.row, self.col]
        elif self.dead:
            self.target = [14, 13]
            return

        # Records the quadrants of each ghost's target
        quads = [0, 0, 0, 0]
        for ghost in game.ghosts:
            # if ghost.target[0] == self.row and ghost.col == self.col:
            #     continue
            if ghost.target[0] <= 15 and ghost.target[1] >= 13:
                quads[0] += 1
            elif ghost.target[0] <= 15 and ghost.target[1] < 13:
                quads[1] += 1
            elif ghost.target[0] > 15 and ghost.target[1] < 13:
                quads[2] += 1
            elif ghost.target[0]> 15 and ghost.target[1] >= 13:
                quads[3] += 1

        # Finds a target that will keep the ghosts dispersed
        while True:
            self.target = [randrange(31), randrange(28)]
            quad = 0
            if self.target[0] <= 15 and self.target[1] >= 13:
                quad = 0
            elif self.target[0] <= 15 and self.target[1] < 13:
                quad = 1
            elif self.target[0] > 15 and self.target[1] < 13:
                quad = 2
            elif self.target[0] > 15 and self.target[1] >= 13:
                quad = 3
            if not gameBoard[self.target[0]][self.target[1]] == 3 and not gameBoard[self.target[0]][self.target[1]] == 4:
                break
            elif quads[quad] == 0:
                break

    def move(self):
        # print(self.target)
        self.lastLoc = [self.row, self.col]
        if self.dir == 0:
            self.row -= self.ghostSpeed
        elif self.dir == 1:
            self.col += self.ghostSpeed
        elif self.dir == 2:
            self.row += self.ghostSpeed
        elif self.dir == 3:
            self.col -= self.ghostSpeed

        # Incase they go through the middle tunnel
        self.col = self.col % len(gameBoard[0])
        if self.col < 0:
            self.col = len(gameBoard[0]) #- 0.5



    def setAttacked(self, isAttacked):
        self.attacked = isAttacked

    def isAttacked(self):
        return self.attacked

    def setDead(self, isDead):
        self.dead = isDead

    def isDead(self):
        return self.dead




def canMove(row, col):
    if col == -1 or col == len(gameBoard[0]):
        return True
    if gameBoard[int(row)][int(col)] != 3:
        return True
    return False

# Reset after death
#EDITSHANKS
def reset():
    # global game,gameBoard
    # gameBoard = copy.deepcopy(originalGameBoard)
    # game.ghosts = [Ghost(14.0, 13, "red", 0), Ghost(17.0, 11, "blue", 1), Ghost(17.0, 13, "pink", 2), Ghost(17.0, 15, "orange", 3)]
    # for ghost in game.ghosts:
    #     ghost.setTarget()
    # game.pacman = Pacman(26.0, 13)
    # game.lives -= 1
    global game,gameBoard
    gameBoard = copy.deepcopy(originalGameBoard)
    game = Game(4, 0)
    game.paused = False
    game.started = True
    # game.render()

# def displayLaunchScreen():
#     # Draw Pacman Title
#     pacmanTitle = ["tile016.png", "tile000.png", "tile448.png", "tile012.png", "tile000.png", "tile013.png"]
#     for i in range(len(pacmanTitle)):
#         letter = pygame.image.load(TextPath + pacmanTitle[i])
#         letter = pygame.transform.scale(letter, (int(square * 4), int(square * 4)))
#         screen.blit(letter, ((2 + 4 * i) * square, 2 * square, square, square))

#     # Draw Character / Nickname
#     characterTitle = [
#         #Character
#         "tile002.png", "tile007.png", "tile000.png", "tile018.png", "tile000.png", "tile002.png", "tile020.png", "tile004.png", "tile018.png",
#         # /
#         "tile015.png", "tile042.png", "tile015.png",
#         # Nickname
#         "tile013.png", "tile008.png", "tile002.png", "tile010.png", "tile013.png", "tile000.png", "tile012.png", "tile004.png"
#     ]
#     for i in range(len(characterTitle)):
#         letter = pygame.image.load(TextPath + characterTitle[i])
#         letter = pygame.transform.scale(letter, (int(square), int(square)))
#         screen.blit(letter, ((4 + i) * square, 10 * square, square, square))

#     #Draw Characters and their Nickname
#     characters = [
#         # Red Ghost
#         [
#             "tile449.png", "tile015.png", "tile107.png", "tile015.png", "tile083.png", "tile071.png", "tile064.png", "tile067.png", "tile078.png", "tile087.png",
#             "tile015.png", "tile015.png", "tile015.png", "tile015.png",
#             "tile108.png", "tile065.png", "tile075.png", "tile072.png", "tile077.png", "tile074.png", "tile089.png", "tile108.png"
#         ],
#         # Pink Ghost
#         [
#             "tile450.png", "tile015.png", "tile363.png", "tile015.png", "tile339.png", "tile336.png", "tile324.png", "tile324.png", "tile323.png", "tile345.png",
#             "tile015.png", "tile015.png", "tile015.png", "tile015.png",
#             "tile364.png", "tile336.png", "tile328.png", "tile333.png", "tile330.png", "tile345.png", "tile364.png"
#         ],
#         # Blue Ghost
#         [
#             "tile452.png", "tile015.png", "tile363.png", "tile015.png", "tile193.png", "tile192.png", "tile211.png", "tile199.png", "tile197.png", "tile213.png", "tile203.png",
#             "tile015.png", "tile015.png", "tile015.png",
#             "tile236.png", "tile200.png", "tile205.png", "tile202.png", "tile217.png", "tile236.png"
#         ],
#         # Orange Ghost
#         [
#             "tile451.png", "tile015.png", "tile363.png", "tile015.png", "tile272.png", "tile270.png", "tile266.png", "tile260.png", "tile281.png",
#             "tile015.png", "tile015.png", "tile015.png", "tile015.png", "tile015.png",
#             "tile300.png", "tile258.png", "tile267.png", "tile281.png", "tile259.png", "tile260.png", "tile300.png"
#         ]
#     ]
#     for i in range(len(characters)):
#         for j in range(len(characters[i])):
#             if j == 0:
#                     letter = pygame.image.load(TextPath + characters[i][j])
#                     letter = pygame.transform.scale(letter, (int(square * spriteRatio), int(square * spriteRatio)))
#                     screen.blit(letter, ((2 + j) * square - square//2, (12 + 2 * i) * square - square//3, square, square))
#             else:
#                 letter = pygame.image.load(TextPath + characters[i][j])
#                 letter = pygame.transform.scale(letter, (int(square), int(square)))
#                 screen.blit(letter, ((2 + j) * square, (12 + 2 * i) * square, square, square))
#     # Draw Pacman and Ghosts
#     event = ["tile449.png", "tile015.png", "tile452.png", "tile015.png",  "tile015.png", "tile448.png", "tile453.png", "tile015.png", "tile015.png", "tile015.png",  "tile453.png"]
#     for i in range(len(event)):
#         character = pygame.image.load(TextPath + event[i])
#         character = pygame.transform.scale(character, (int(square * 2), int(square * 2)))
#         screen.blit(character, ((4 + i * 2) * square, 24 * square, square, square))
#     # Draw PlatForm from Pacman and Ghosts
#     wall = ["tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png", "tile454.png"]
#     for i in range(len(wall)):
#         platform = pygame.image.load(TextPath + wall[i])
#         platform = pygame.transform.scale(platform, (int(square * 2), int(square * 2)))
#         screen.blit(platform, ((i * 2) * square, 26 * square, square, square))
#     # Credit myself
#     credit = ["tile003.png", "tile004.png", "tile022.png", "tile008.png", "tile013.png", "tile015.png", "tile011.png", "tile004.png", "tile000.png", "tile012.png", "tile025.png", "tile015.png", "tile418.png", "tile416.png", "tile418.png", "tile416.png"]
#     for i in range(len(credit)):
#         letter = pygame.image.load(TextPath + credit[i])
#         letter = pygame.transform.scale(letter, (int(square), int(square)))
#         screen.blit(letter, ((6 + i) * square, 30 * square, square, square))
#     # Press Space to Play
#     instructions = ["tile016.png", "tile018.png", "tile004.png", "tile019.png", "tile019.png", "tile015.png", "tile019.png", "tile016.png", "tile000.png", "tile002.png", "tile004.png", "tile015.png", "tile020.png", "tile014.png", "tile015.png", "tile016.png", "tile011.png", "tile000.png", "tile025.png"]
#     for i in range(len(instructions)):
#         letter = pygame.image.load(TextPath + instructions[i])
#         letter = pygame.transform.scale(letter, (int(square), int(square)))
#         screen.blit(letter, ((4.5 + i) * square, 35 * square - 10, square, square))

#     pygame.display.update()



####### PACMAN GAME IMPLEMENTATION, NOT MY CODE ######## - But was adapted for my use!



####### MY FUNCTIONS ########
def print_graphs(DQN_training_information,DDQN_training_information):
    #Q_episode,Q_episode_reward,Q_win_rate,Q_random_actions_taken,Q_mean_reward = zip(*Q_learning_training_information)
    DQN_episode,DQN_episode_reward,DQN_win_rate,DQN_random_actions_taken,DQN_mean_reward = zip(*DQN_training_information)
    DDQN_episode,DDQN_episode_reward,DDQN_win_rate,DDQN_random_actions_taken,DDQN_mean_reward = zip(*DDQN_training_information)
    # Plot episode reward
    # Plot DQN rewards in blue
    plt.figure(figsize=(10, 5))
    plt.plot(DQN_episode, DQN_episode_reward, label='DQN', color='blue')
    # Plot DDQN rewards in green
    plt.plot(DDQN_episode, DDQN_episode_reward, label='DDQN', color='green')
    # # Plot Q learning reward in red
    # plt.plot(Q_episode, Q_episode_reward, label='Tabular RL', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('DQN vs DDQN Performance Over 25000 Episodes')
    plt.savefig('episode_rewards_plot_DQNvsDDQN.jpg')

    # Plot win rate
    # Plot DQN Winrate in blue
    plt.figure(figsize=(10, 5))
    plt.plot(DQN_episode, DQN_win_rate, label='DQN', color='blue')

    # Plot DDQN Winrate in green
    plt.plot(DDQN_episode, DDQN_win_rate, label='DDQN', color='green')

    # # Plot Q learning Winrate in red
    # plt.plot(Q_episode, Q_win_rate, label='Tabular RL', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title('DQN vs DDQN Win rate Over 25000 Episodes')
    plt.savefig('episode_winrate_plot_DQNvsDDQN.jpg')


    # Plot random actions per episode
    # Plot DQN random actions in blue
    plt.figure(figsize=(10, 5))
    plt.plot(DQN_episode, DQN_random_actions_taken, label='DQN', color='blue')

    # Plot DDQN random actions in green
    plt.plot(DDQN_episode, DDQN_random_actions_taken, label='DDQN', color='green')

    # # Plot Q learning random actions in red
    # plt.plot(Q_episode, Q_random_actions_taken, label='Tabular RL', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Average percentage of random actions (%)')
    plt.legend()
    plt.title('DQN vs DDQN, Percentage of Random Actions Over 25000 Episodes')
    plt.savefig('episode_random_actions_plot_DQNvsDDQN.jpg')

    # Plot mean episode reward
    # Plot DQN rewards in blue
    plt.figure(figsize=(10, 5))
    plt.plot(DQN_episode, DQN_mean_reward, label='DQN', color='blue')

    # Plot DDQN rewards in green
    plt.plot(DDQN_episode, DDQN_mean_reward, label='DDQN', color='green')

    # # Plot Q learning random actions in red
    # plt.plot(Q_episode, Q_mean_reward, label='Tabular RL', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.legend()
    plt.title('DQN vs DDQN vs Performance Over 25000 Episodes')
    plt.savefig('average_episode_rewards_plot_DQNvsDDQN.jpg')


def print_graph_1(DQN_training_information):
    #Q_episode,Q_episode_reward,Q_win_rate,Q_random_actions_taken,Q_mean_reward = zip(*Q_learning_training_information)
    DQN_episode,DQN_episode_reward,DQN_win_rate,DQN_random_actions_taken,DQN_mean_reward = zip(*DQN_training_information)
    # Plot episode reward
    # Plot DQN rewards in blue
    plt.figure(figsize=(10, 5))
    plt.plot(DQN_episode, DQN_episode_reward, label='DQN', color='blue')
    # Plot DDQN rewards in green
  
    # # Plot Q learning reward in red
    # plt.plot(Q_episode, Q_episode_reward, label='Tabular RL', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('DQN MDP Stacked state matrix')
    plt.savefig('reward_unfinished.jpg')

    # Plot win rate
    # Plot DQN Winrate in blue
    plt.figure(figsize=(10, 5))
    plt.plot(DQN_episode, DQN_win_rate, label='DQN', color='blue')

    # Plot DDQN Winrate in green


    # # Plot Q learning Winrate in red
    # plt.plot(Q_episode, Q_win_rate, label='Tabular RL', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title('DQN MDP Stacked state matrix')
    plt.savefig('win_rate_unfinished.jpg')


    # Plot random actions per episode
    # Plot DQN random actions in blue
    plt.figure(figsize=(10, 5))
    plt.plot(DQN_episode, DQN_random_actions_taken, label='DQN', color='blue')

    # Plot DDQN random actions in green


    # # Plot Q learning random actions in red
    # plt.plot(Q_episode, Q_random_actions_taken, label='Tabular RL', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Average percentage of random actions (%)')
    plt.legend()
    plt.title('DQN MDP Stacked state matrix')
    plt.savefig('random_actions_unfinished.jpg')

    # Plot mean episode reward
    # Plot DQN rewards in blue
    plt.figure(figsize=(10, 5))
    plt.plot(DQN_episode, DQN_mean_reward, label='DQN', color='blue')

    # Plot DDQN rewards in green
  

    # # Plot Q learning random actions in red
    # plt.plot(Q_episode, Q_mean_reward, label='Tabular RL', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.legend()
    plt.title('DQN MDP Stacked state matrix')
    plt.savefig('average_reward_unfinished.jpg')

def average_training_information(all_training_information):

    num_training_runs = len(all_training_information)
    num_episodes = len(all_training_information[0])
    average_training_info = []
    # Calculate the average episode reward for each episode
    for episode in range(num_episodes):
        episode_rewards = [run[episode][1] for run in all_training_information]
        average_reward = sum(episode_rewards) / num_training_runs
        
        score = [run[episode][2] for run in all_training_information]
        average_score =  sum(score) / num_training_runs

        mean_reward = [run[episode][3] for run in all_training_information]
        average_mean_reward = sum(mean_reward) / num_training_runs

        mean_score = [run[episode][4] for run in all_training_information]
        average_mean_score = sum(mean_score) / num_training_runs

        average_training_info.append((episode,average_reward,average_score,average_mean_reward,average_mean_score))
    
    return average_training_info

def send_email():
    PORT = 587
    EMAIL_SERVER = "smtp-mail.outlook.com"
    sender_email = "shanks.notify@outlook.com"
    password_email = "P@ssmes123"
    reciever_email = "shanks.notify@gmail.com"
    current_time = datetime.now().time()
    
    msg = EmailMessage()
    msg["Subject"] = "The run you have started is complete!"
    msg["From"] = formataddr(("Run Complete",f"{sender_email}"))
    msg["To"] = reciever_email
    msg["BCC"] = sender_email

    msg.set_content(
        f"The run that you have started has completed at {current_time}"
    )

    with smtplib.SMTP(EMAIL_SERVER,PORT) as server:
        server.starttls()
        server.login(sender_email,password_email)
        server.sendmail(sender_email,"shanks.notify@gmail.com", msg.as_string())

def save_file(info,filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(info,file)

def load_file(filepath):
    with open(filepath, 'rb') as file:
        info = pickle.load(file)

    return info

def pause(time):
    cur = 0
    while not cur == time:
        cur += 1

def get_state_with_pacman_and_ghosts():
    state = np.array(gameBoard.copy())
    for ghost in game.ghosts:
        if game.ghostsAttacked == True:
            state[ghost.row,ghost.col] = dp.SCARED_GHOSTS
        else:
            state[ghost.row,ghost.col] = dp.ENEMY

    state[game.pacman.row,game.pacman.col] = dp.AGENT

    return state


def populate_replay_buffer(a:dpa.DQNAgent):
    reset()
    num_random_samples = round(0.2 * a.rb.buffer_size)
    st = get_state_with_pacman_and_ghosts()
    st_stacked  = a.get_MDP_matrix(st)
    for i in range(num_random_samples):
        prev_rew = game.score
        at = a.pick_random_action(st_stacked)
        game.pacman.newDir = at
        game.update()
        new_st = get_state_with_pacman_and_ghosts()
        new_st_stacked = a.get_MDP_matrix(new_st)
        new_rev  = game.score
        rt = new_rev - prev_rew
        done = game.gameOver
        if done and game.killed_by_ghosts:
            rt -= 400
            print('Killed by Ghost')
        elif done and not game.killed_by_ghosts:
            rt += 400
            print('Won game!')
        transition = dp.Sarsd(st_stacked,at,rt,new_st_stacked,done)
        a.rb.push(transition)
        st_stacked = new_st_stacked
        if done:
            print('done')
            reset()
            st = get_state_with_pacman_and_ghosts()
            st_stacked  = a.get_MDP_matrix(st)
    
    print("finished Populating the replay buffer")


def episode_training(a:dpa.DQNAgent,training_Params:dp.TrainingParams,DeepRLAlg,EPS,epconst):
    populate_replay_buffer(a)

    eps_end = training_Params.epsilon_end
    eps_start = training_Params.epsilon_start
    eps_decay = training_Params.epsilon_decay
  
    epsilon = epconst

    episodes = training_Params.training_episodes
    total_steps = 0
    episode_steps = 0
    episode_reward = 0.0
    average_reward_buffer = deque([0.0],50)
    average_score_buffer = deque([0.0],50)
    average_random_actions_buffer= deque([0,0],50)
    training_information = []
    loss = []
    wins = 0
    random_actions_taken = 0
    win_rate = 0

    for i in range(episodes):

        reset()
        st = get_state_with_pacman_and_ghosts()
        st_stacked = a.get_MDP_matrix(st)
        done = False

        while not done:

            prev_rew = game.score
            sample = random.random()


            if EPS == dp.CONST_EP:
                epsilon = epconst
            elif EPS == dp.CURVE_EP:
                epsilon = eps_end + (eps_start-eps_end) * math.exp(-1 * total_steps/eps_decay)
            elif EPS == dp.LINEAR_EP:
                epsilon = eps_end + (eps_start-eps_end) * (1 - min(1,total_steps/eps_decay))
            else:
                epsilon = 0.1
                print("Something went wrong epsilon is default 0.1")


            if sample > epsilon:
                at = a.pick_dqn_action(st_stacked)
            else:
                at = a.pick_random_action(st_stacked)
                random_actions_taken += 1

            game.pacman.newDir = at
            game.update()
            new_st = get_state_with_pacman_and_ghosts()
            new_st_stacked = a.get_MDP_matrix(new_st)
            new_rev  = game.score   
            rt = new_rev - prev_rew
            done = game.gameOver
            if done and game.killed_by_ghosts:
                rt -= 400
                print('Killed by Ghost')
            elif done and not game.killed_by_ghosts:
                wins+= 1
                rt += 400
                print('Won game!')
            transition = dp.Sarsd(st_stacked,at,rt,new_st_stacked,done)
            a.rb.push(transition)

            if DeepRLAlg == dp.DQN_ALG:
                step_loss = a.optimize_model_DQN()
            elif DeepRLAlg == dp.DDQN_ALG:
                step_loss = a.optimize_model_DDQN()
            elif DeepRLAlg == dp.DQN_ALG_PR:
                step_loss = a.optimize_model_DQN_PR(epsilon,total_steps)
            else:
                print('Unknown algorithm to use, no optimising is being done.')
            
            
            st_stacked = new_st_stacked
            episode_reward+= rt
            total_steps+= 1
            episode_steps+=1
            loss.append((total_steps,step_loss))
            if total_steps % a.target_update_hz == 0:
                a.target_net.load_state_dict(a.policy_net.state_dict())

            # MDP means normal MDP with no one hot encoding
            # Stacked means stacked matrix but not MDP
            # StackedMDP means MDP and stacked

            if total_steps % 10000 == 0:

                print("Updating files")
                if a.target_update_hz == 100:
                    torch.save(a.policy_net.state_dict(),'policy_DDQN_3000_2D_MDP_targ_100_25102023.pth' )
                elif a.target_update_hz == 10000:
                    torch.save(a.policy_net.state_dict(),'policy_DDQN_3000_2D_MDP_targ_10000_25102023.pth' )
                else:
                     torch.save(a.policy_net.state_dict(),'policy_DDQN_3000_2D_MDP_unknown_25102023.pth' )
                
                # if DeepRLAlg == dp.DQN_ALG:
                #     torch.save(agent.policy_net.state_dict(),'policy_DQN_3000_2D_MDP_23102023.pth' ) #Algorithm_Episodes_Epsilon_variedtrainingparam_date
                # elif DeepRLAlg == dp.DDQN_ALG:
                #     torch.save(agent.policy_net.state_dict(),'policy_DDQN_3000_2D_MDP_23102023.pth' ) #Algorithm_Episodes_Epsilon_variedtrainingparam_date
                # elif DeepRLAlg == dp.DQN_ALG_PR:
                #     torch.save(agent.policy_net.state_dict(),'policy_DQN_PR_3000_2D_MDP_23102023.pth' )
                # else:
                #     print('Unknown algorithm to use, no optimising is being done.')


       


      

        win_rate = win_rate + wins 

        print(f'Episode : {i}')
        print(f'Number of random actions taken: {random_actions_taken}')
        print(f'Number of Wins : {win_rate} %')
        print(f'Score for episode: {episode_reward}')
        print(f'In game score for episode: {game.score}')
        # print(f'Episode threshold: {eps_threshold}')
        print(f'total steps: {total_steps}')

        average_reward_buffer.append(episode_reward)
        average_score_buffer.append(game.score)
        average_random_actions_buffer.append((random_actions_taken/episode_steps) * 100)
        mean_reward = np.mean(average_reward_buffer)
        mean_score = np.mean(average_score_buffer)
        training_information.append((i,episode_reward,game.score,mean_reward,mean_score)) 

        episode_reward = 0.0
        episode_steps = 0
        random_actions_taken = 0


        if (i+1) % 10 == 0:
            save_file(training_information,'unfinished_training.pkl')
            save_file(loss,'loss.pkl')




    
    return training_information


####### MY FUNCTIONS ########


####### TRAINING #######
all_DQN_training_information = []
start_time = time.time()

for i in range(3):

    game = Game(4, 0)
    ghostsafeArea = [15, 13] # The location the ghosts escape to when attacked
    ghostGate = [[15, 13], [15, 14]]

    running = True
    onLaunchScreen = False
    game.paused = False
    game.started = True

    training_params = dp.TrainingParams(dp.EPSILON_START,dp.EPSILON_DECAY,dp.EPSILON_END,dp.EPISODES)
    character_params = dp.CharacterParams(dp.PATH,dp.GOAL,dp.WALL,dp.GHOST_SAFE_SPACE,dp.SPECIAL_PELLETS_V1,dp.SPECIAL_PELLETS_V2,dp.ENEMY,dp.SCARED_GHOSTS,dp.AGENT)
    deepQ_params = dp.DeepQParams(dp.GAMMA,dp.BATCH_SIZE,dp.BUFFER_SIZE,dp.MIN_REPLAY_SIZE,100,dp.OPTIMIZER_LR)
    rb = dpa.ReplayBuffer(deepQ_params)
    agent = dpa.DQNAgent(character_params,deepQ_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent.assign_device(device)

    maze = agent.get_MDP_matrix(get_state_with_pacman_and_ghosts())
    nd,nr,nc = maze.shape
    print(f"depth: {nd}, rows: {nr}, columns: {nc}")

    policy_net = dpa.CNN2DDQN(nd,nr,nc,4,device)
    target_net = dpa.CNN2DDQN(nd,nr,nc,4,device)
    target_net.load_state_dict(policy_net.state_dict())
    agent.assign_policy_net(policy_net)
    agent.assign_target_net(target_net)
    agent.assign_replay_buffer(rb)
    agent.assign_optimizer()
    np.set_printoptions(threshold=np.inf)

    DQN_training_information = episode_training(agent,training_params,dp.DDQN_ALG,dp.CONST_EP,0.1)
    all_DQN_training_information.append(DQN_training_information)

end_time = time.time()
elapsed_time_DQN = end_time - start_time
start_time = time.time()

average_all_DQN_training_info = average_training_information(all_DQN_training_information)
save_file(average_all_DQN_training_info,'DDQN_25_OCT_3000_targ_100.pkl')


print("Done training")
print(f"Training time DDQN Target update 100 : {elapsed_time_DQN}")
send_email()

####### TRAINING #######


    
