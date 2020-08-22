import json

import numpy as np

import constants
from color_text import ColorText


def print_map(obs):
    width = 8
    print(f"Steps: {obs.stepCount}")
    # print(f"Energy ", end='')
    # for i in range(4):
    #     print(f"({i}):{obs.players[i]['energy']:5}\t", end='')
    # print()

    # print(f"Gold   ", end='')
    # for i in range(4):
    #     print(f"({i}):{obs.players[i]['score']:5}\t", end='')
    # print()

    for i in range(constants.N_ROWS):
        for j in range(constants.N_COLS):
            players = ""
            type, value = None, None
            for cell in obs.mapInfo.obstacles:
                if j == cell["posx"] and i == cell["posy"]:
                    type, value = cell["type"], cell["value"]

            text_color = ColorText.CBLACK
            if type is None:
                type = 4


            for k, player in enumerate(obs.players):
                if "status" in player and player["status"] == 0 and j == player["posx"] and i == player["posy"]:
                    if k != 0:
                        players += " "
                    players += str(k)
                    text_color = ColorText.CWHITE2

            color = ColorText.CWHITEBG

            if type == 1:
                color = ColorText.CGREENBG
            elif type == 2:
                color = ColorText.CGREYBG
            elif type == 3:
                color = ColorText.CBLUEBG
            elif type == 4:
                color = ColorText.CYELLOWBG

            print(f"{color}{text_color}{players:{width}}{ColorText.CEND}", end="")
        print()

        for j in range(constants.N_COLS):
            type, value = None, None
            for cell in obs.mapInfo.obstacles:
                if j == cell["posx"] and i == cell["posy"]:
                    type, value = cell["type"], cell["value"]

            text_color = ColorText.CBLACK
            if type is None:
                value = obs.mapInfo.gold_amount(j, i)
                type = 4
            elif type != constants.Obstacle.SWAMP.value:
                value = ""
            color = ColorText.CWHITEBG

            if type == 1:
                color = ColorText.CGREENBG
            elif type == 2:
                color = ColorText.CGREYBG
            elif type == 3:
                color = ColorText.CBLUEBG
            elif type == 4:
                color = ColorText.CYELLOWBG

            print(f"{color}{text_color}{str(value):{width}}{ColorText.CEND}", end="")
        print()
    print()


def inside_map(row, col):
    return 0 <= row < 9 and 0 <= col < 21
