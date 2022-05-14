#!/usr/bin/env python3
# encoding: utf-8

import random
import sys
import os
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

models = "qtable.npy"

SW = 288
SH = 512
FPS = 36

BASEY = SH * 0.8
IMAGES, SOUNDS = {}, {}
pygame.font.init()
SCREEN = pygame.display.set_mode((SW, SH))
Font = pygame.font.SysFont("comicsans", 20)
BIRD = 'imgs/bird1.png'
BG = 'imgs/bg.png'
PIPE = 'imgs/pipe.png'
PIPE_DISTANCE = SW + 10
Q = np.load(models) if os.path.exists(models) else np.zeros((8, 20, 2), dtype=float) # 8: birdxpos 20: birdypos 2: 1跳0不跳


def static():
    birdxpos = int(SW/5)
    birdypos = int(SH/2)
    basex = 0
    text1 = Font.render("› HUMAN", 1, (0, 0, 0))
    text2 = Font.render("  AI", 1, (255, 255, 255))
    aiplayer = False
    while (True):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == KEYDOWN:
                if event.key == K_SPACE or event.key == K_RETURN:
                    return aiplayer
                elif event.key == K_UP:
                    aiplayer = False
                    text1 = Font.render("› HUMAN", 1, (0, 0, 0))
                    text2 = Font.render("  AI", 1, (255, 255, 255))
                elif event.key == K_DOWN:
                    aiplayer = True
                    text1 = Font.render("  HUMAN", 1, (255, 255, 255))
                    text2 = Font.render("› AI", 1, (0, 0, 0))
            else:
                SCREEN.blit(IMAGES['background'], (0, 0))
                SCREEN.blit(IMAGES['bird'], (birdxpos, birdypos))
                SCREEN.blit(IMAGES['base'], (basex, BASEY))
                # SCREEN.blit(Font.render(str(birdypos), 1, (255, 255, 255)), (int(SW/6), 10))
                text1_rect = text1.get_rect()
                text2_rect = text2.get_rect()
                SCREEN.blit(
                    text1, (SW/2 - (text1_rect[2]/2), SH/2 - (text1_rect[3]/2) - 10))
                SCREEN.blit(
                    text2, (SW/2 - (text2_rect[2]/2), SH/2 + (text2_rect[3]/2) + 10))
                pygame.display.update()
                FPSCLOCK.tick(FPS)
    return aiplayer


def game_start(generation, x, y, is_ai_player):
    score = 0
    birdxpos = int(SW/5)
    birdypos = int(SH/2)
    basex1 = 0
    basex2 = SW

    bgx1 = 0
    bgx2 = IMAGES['background'].get_width()
    newPipe1 = get_new_pipe()
    up_pipes = [
        {'x': SW + 10, 'y': newPipe1[0]['y']}
    ]

    bttm_pipes = [
        {'x': SW + 10, 'y': newPipe1[1]['y']}
    ]

    pipeVelx = -4

    birdyvel = -9
    birdymaxvel = 10
    birdyvelmin = -8
    birdyacc = 1

    playerFlapAccv = -9
    playerFlapped = False

    while(True):

        x_prev, y_prev = convert(birdxpos, birdypos, bttm_pipes)
        jump = ai_player(x_prev, y_prev) if is_ai_player else human_player()

        for event in pygame.event.get():
            if event.type == QUIT:
                np.save(models, Q, allow_pickle=False)
                plt.scatter(counts, scores)
                plt.xlabel("GENERATIONS")
                plt.ylabel("SCORES")
                plt.title("Flappy Birds : AI")
                plt.show()
                pygame.quit()
                sys.exit()

        if jump:
            if birdypos > 0:
                birdyvel = playerFlapAccv
                playerFlapped = True

        playerMidPos = birdxpos + IMAGES['bird'].get_width()/2
        for pipe in up_pipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width()/2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                if not is_ai_player: SOUNDS["point"].play()

        if birdyvel < birdymaxvel and not playerFlapped:
            birdyvel += birdyacc

        if playerFlapped:
            playerFlapped = False

        playerHeight = IMAGES['bird'].get_height()

        birdypos = birdypos + min(birdyvel, BASEY - birdypos - playerHeight)

        for upperPipe, lowerPipe in zip(up_pipes, bttm_pipes):
            upperPipe['x'] += pipeVelx
            lowerPipe['x'] += pipeVelx

        if (0 <= up_pipes[0]['x'] <= 4):
            newPipe = get_new_pipe()
            up_pipes.append(newPipe[0])
            bttm_pipes.append(newPipe[1])

        if(up_pipes[0]['x'] < -IMAGES['pipe'][0].get_width()):
            up_pipes.pop(0)
            bttm_pipes.pop(0)
        basex1 -= 4
        basex2 -= 4
        if(basex1 <= -IMAGES['base'].get_width()):
            basex1 = basex2
            basex2 = basex1 + IMAGES['base'].get_width()

        bgx1 -= 2
        bgx2 -= 2
        if(bgx1 <= -IMAGES['background'].get_width()):
            bgx1 = bgx2
            bgx2 = bgx1 + IMAGES['background'].get_width()
        crashTest = Collision(birdxpos, birdypos, up_pipes, bttm_pipes)
        x_new, y_new = convert(birdxpos, birdypos, bttm_pipes)
        if crashTest:
            reward = -1000
            Q_update(x_prev, y_prev, jump, reward, x_new, y_new)
            if not is_ai_player: SOUNDS["hit"].play()
            return score

        reward = 1

        Q_update(x_prev, y_prev, jump, reward, x_new, y_new)

        SCREEN.blit(IMAGES['background'], (bgx1, 0))
        SCREEN.blit(IMAGES['background'], (bgx2, 0))
        for upperPipe, lowerPipe in zip(up_pipes, bttm_pipes):
            SCREEN.blit(IMAGES['pipe'][0], (upperPipe['x'], upperPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lowerPipe['x'], lowerPipe['y']))
        SCREEN.blit(IMAGES['base'], (basex1, BASEY))
        SCREEN.blit(IMAGES['base'], (basex2, BASEY))
        text1 = Font.render("Score: " + str(score), 1, (255, 255, 255))
        text2 = Font.render(
            "Generation: " + str(generation), 1, (255, 255, 255))
        SCREEN.blit(text1, (SW - 10 - text1.get_width(), 0))
        SCREEN.blit(text2, (0, 0))
        # Debug
        # text3 = Font.render('birdxpos:' + str(x_new), 1, (255, 255, 255))
        # text4 = Font.render('birdypos:' + str(y_new), 1, (255, 255, 255))
        # SCREEN.blit(text3, (0, text2.get_rect()[3] + 10))
        # SCREEN.blit(text4, (0, text2.get_rect()[3] + text3.get_rect()[3] + 20))
        SCREEN.blit(IMAGES['bird'], (birdxpos, birdypos))

        pygame.display.update()
        FPSCLOCK.tick(0 if is_ai_player else FPS)


def Collision(birdxpos, birdypos, up_pipes, bttm_pipes):
    if (birdypos >= BASEY - IMAGES['bird'].get_height() or birdypos < 0):
        return True
    for pipe in up_pipes:
        pipeHeight = IMAGES['pipe'][0].get_height()
        if(birdypos < pipeHeight + pipe['y'] and abs(birdxpos - pipe['x']) < IMAGES['pipe'][0].get_width()):
            return True

    for pipe in bttm_pipes:
        if (birdypos + IMAGES['bird'].get_height() > pipe['y'] and abs(birdxpos - pipe['x']) < IMAGES['pipe'][0].get_width()):
            return True
    return False


def human_player():
    jump = False
    for event in pygame.event.get():
        if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
            jump = True
    return jump


def ai_player(x, y):
    return Q[x][y][1] > Q[x][y][0]

def get_new_pipe():
    pipeHeight = IMAGES['pipe'][1].get_height()
    gap = int(SH/4)
    y2 = int(random.randrange(gap, int(BASEY)))
    pipex = int(SW+10)
    y1 = int(pipeHeight - y2 + gap)

    pipe = [
        {'x': pipex, 'y': -y1},
        {'x': pipex, 'y': y2}
    ]
    return pipe

def convert(birdxpos, birdypos, bttm_pipes):
    x = min(SW, max(bttm_pipes[0]['x'], 0))
    y = bttm_pipes[0]['y'] - birdypos
    if(y < 0):
        y = abs(y)+SH
    return int(x/40), int(y/40)


def Q_update(x_prev, y_prev, jump, reward, x_new, y_new):
    if jump:
        Q[x_prev][y_prev][1] = 0.3 * Q[x_prev][y_prev][1] + \
            (0.7)*(reward + max(Q[x_new][y_new][0], Q[x_new][y_new][1]))
    else:
        Q[x_prev][y_prev][0] = 0.3 * Q[x_prev][y_prev][0] + \
            (0.7)*(reward + max(Q[x_new][y_new][0], Q[x_new][y_new][1]))


if __name__ == "__main__":

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    pygame.display.set_caption("Flappy Birds")

    IMAGES['base'] = pygame.image.load('imgs/base.png').convert_alpha()
    IMAGES['pipe'] = (pygame.transform.rotate(pygame.image.load(
        PIPE).convert_alpha(), 180), pygame.image.load(PIPE).convert_alpha())
    IMAGES['background'] = pygame.image.load(BG).convert()
    IMAGES['bird'] = pygame.image.load(BIRD).convert_alpha()

    # sounds
    if "win" in sys.platform:
        soundExt = ".wav"
    else:
        soundExt = ".ogg"

    SOUNDS["hit"] = pygame.mixer.Sound("assets/hit" + soundExt)
    SOUNDS["point"] = pygame.mixer.Sound("assets/point" + soundExt)

    generation = 1
    is_ai_player = static()
    counts = []
    scores = []
    while(True):
        score = game_start(generation, counts, scores,
                           is_ai_player)
        if (score == -1):
            break
        counts.append(generation)
        scores.append(score)
        generation += 1
