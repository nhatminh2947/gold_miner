import json
import math
import os
from random import randrange

import numpy as np

import constants
import utils
from . import PlayerInfo, UserMatch, ObstacleInfo, GoldInfo


class StepState:
    def __init__(self):
        self.players = []
        self.golds = []
        self.changedObstacles = []

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class GameSocket:
    bog_energy_chain = {-5: -20, -20: -40, -40: -100, -100: -100}

    def __init__(self, host, port):
        self.stepCount = 0
        self.maxStep = 0
        self.mapdir = "/home/lucius/working/projects/gold_miner/resources/original_maps"  # where to load all pre-defined maps
        self.mapid = ""
        self.userMatch = UserMatch()
        self.users = [PlayerInfo(i) for i in range(4)]
        self.stepState = StepState()
        self.maps = {}  # key: map file name, value: file content
        self.map = []  # running map info: 0->Land, -1->Forest, -2->Trap, -3:Swamp, >0:Gold
        self.energyOnMap = []  # self.energyOnMap[x][y]: <0, amount of energy which player will consume if it move into (x,y)
        self.E = 50
        self.resetFlag = True
        self.craftUsers = []  # players that craft at current step - for calculating amount of gold
        self.craftMap = {}  # cells that players craft at current step, key: x_y, value: number of players that craft at (x,y)

        # def init_bots(self):
        #     self.bots = [Bot1(2), Bot2(3), Bot3(4)]  # use bot1(id=2), bot2(id=3), bot3(id=4)
        #     for (bot) in self.bots:  # at the beginning, all bots will have same position, energy as player
        #         bot.info.posx = self.user.posx
        #         bot.info.posy = self.user.posy
        #         bot.info.energy = self.user.energy
        #         bot.info.lastAction = -1
        #         bot.info.status = PlayerInfo.STATUS_PLAYING
        #         bot.info.score = 0
        #         self.stepState.players.append(bot.info)
        #     self.userMatch.gameinfo.numberOfPlayers = len(self.stepState.players)
        #     print("numberOfPlayers: ", self.userMatch.gameinfo.numberOfPlayers)

    def reset(self, requests):  # load new game by given request: [map id (filename), posx, posy, initial energy]
        # load new map
        self.reset_map(requests[0])
        self.userMatch.posx = int(requests[1])
        self.userMatch.posy = int(requests[2])
        self.userMatch.energy = int(requests[3])
        self.userMatch.gameinfo.steps = int(requests[4])
        self.maxStep = self.userMatch.gameinfo.steps

        # # init data for players
        # self.user.posx = self.userMatch.posx  # in
        # self.user.posy = self.userMatch.posy
        # self.user.energy = self.userMatch.energy
        # self.user.status = PlayerInfo.STATUS_PLAYING
        # self.user.score = 0
        for user in self.users:
            user.reset(self.userMatch.posx, self.userMatch.posy, self.userMatch.energy)
        self.stepState.players = self.users
        self.E = self.userMatch.energy
        self.resetFlag = True
        self.userMatch.gameinfo.numberOfPlayers = len(self.stepState.players)

        # self.init_bots()
        self.stepCount = 0

    def reset_map(self, id):  # load map info
        self.mapId = "generate" if np.random.uniform() < 0.1 else id
        if self.mapId == "generate":
            self.maps["generate"] = utils.generate_map()
        self.map = json.loads(self.maps[self.mapId])
        self.userMatch = self.map_info(self.map)
        self.stepState.golds = self.userMatch.gameinfo.golds
        self.map = json.loads(self.maps[self.mapId])
        self.energyOnMap = json.loads(self.maps[self.mapId])

        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                if self.map[x][y] > 0:  # gold
                    self.energyOnMap[x][y] = -4
                else:  # obstacles
                    self.energyOnMap[x][y] = ObstacleInfo.types[self.map[x][y]]

    def connect(self):  # simulate player's connect request
        # print("Connected to server.")
        # load all pre-defined maps from mapDir
        path = os.getcwd()

        # print("path:", path)
        for filename in os.listdir(self.mapdir):
            # print("Found: " + filename)
            with open(os.path.join(self.mapdir, filename), 'r') as f:
                self.maps[filename] = f.read()

    def map_info(self, map):  # get map info
        # print(map)
        userMatch = UserMatch()
        userMatch.gameinfo.height = len(map)
        userMatch.gameinfo.width = len(map[0])
        i = 0
        max_gold_num = 10000
        while i < len(map):
            j = 0
            while j < len(map[i]):
                if map[i][j] > 0:  # gold
                    g = GoldInfo()
                    g.posx = j
                    g.posy = i
                    g.amount = map[i][j]
                    userMatch.gameinfo.golds.append(g)
                else:  # obstacles
                    o = ObstacleInfo()
                    o.posx = j
                    o.posy = i
                    o.type = -map[i][j]
                    o.value = ObstacleInfo.types[map[i][j]]
                    userMatch.gameinfo.obstacles.append(o)
                j += 1
            i += 1
        return userMatch

    def receive(self):  # send data to player (simulate player's receive request)
        if self.resetFlag:  # for the first time -> send game info
            self.resetFlag = False
            data = self.userMatch.to_json()

            return data
        else:  # send step state
            self.stepCount = self.stepCount + 1
            if self.stepCount >= self.maxStep:
                for player in self.stepState.players:
                    player.status = constants.Status.STATUS_STOP_END_STEP.value
            data = self.stepState.to_json()

            return data

    def get_actions(self, message):
        actions = [int(action) for action in message.split(',')]

        # if len(actions) != 4:
        #     raise Exception("Not enough actions {}".format(actions))

        return actions

    def send(self, message):  # receive message from player (simulate send request from player)
        if "map" in message:  # reset game
            requests = message.split(",")
            self.reset(requests)
        else:
            actions = self.get_actions(message)  # send 4 action of player
            self.resetFlag = False
            self.stepState.changedObstacles = []
            # print("Action = ", action)
            self.craftUsers = []
            # self.step_action(self.user, action)

            for user, action in zip(self.users, actions):
                if user.status == constants.Status.STATUS_PLAYING.value:
                    user.lastAction = action
                    self.step_action(user, action)

            # for bot in self.bots:
            #     if bot.info.status == PlayerInfo.STATUS_PLAYING:
            #         action = bot.next_action()
            #         bot.info.lastAction = action
            #         # print("Bot Action: ", action)
            #         self.step_action(bot.info, action)
            # for action in actions:
            #     if bot.info.status == PlayerInfo.STATUS_PLAYING:
            #         self.step_action(bot.info, action)
            self.action_5_craft()
            for c in self.stepState.changedObstacles:
                self.map[c["posy"]][c["posx"]] = -c["type"]
                self.energyOnMap[c["posy"]][c["posx"]] = c["value"]

    def step_action(self, user, action):
        switcher = {
            0: self.action_0_left,
            1: self.action_1_right,
            2: self.action_2_up,
            3: self.action_3_down,
            4: self.action_4_free,
            5: self.action_5_craft_pre
        }
        func = switcher.get(action, self.invalidAction)
        func(user)

    def action_5_craft_pre(self, user):  # collect players who craft at current step
        user.freeCount = 0
        if self.map[user.posy][user.posx] <= 0:  # craft at the non-gold cell
            user.energy -= 10
            if user.energy <= 0:
                user.status = constants.Status.STATUS_ELIMINATED_OUT_OF_ENERGY.value
                user.lastAction = 6  # eliminated
        else:
            user.energy -= 5
            if user.energy > 0:
                self.craftUsers.append(user)
                key = str(user.posx) + "_" + str(user.posy)
                if key in self.craftMap:
                    count = self.craftMap[key]
                    self.craftMap[key] = count + 1
                else:
                    self.craftMap[key] = 1
            else:
                user.status = constants.Status.STATUS_ELIMINATED_OUT_OF_ENERGY.value
                user.lastAction = 6  # eliminated

    def action_0_left(self, user):  # user go left
        user.freeCount = 0
        user.posx = user.posx - 1
        if user.posx < 0:
            user.status = constants.Status.STATUS_ELIMINATED_WENT_OUT_MAP.value
            user.lastAction = 6  # eliminated
        else:
            self.go_to_pos(user)

    def action_1_right(self, user):  # user go right
        user.freeCount = 0
        user.posx = user.posx + 1
        if user.posx >= self.userMatch.gameinfo.width:
            user.status = constants.Status.STATUS_ELIMINATED_WENT_OUT_MAP.value
            user.lastAction = 6  # eliminated
        else:
            self.go_to_pos(user)

    def action_2_up(self, user):  # user go up
        user.freeCount = 0
        user.posy = user.posy - 1
        if user.posy < 0:
            user.status = constants.Status.STATUS_ELIMINATED_WENT_OUT_MAP.value
            user.lastAction = 6  # eliminated
        else:
            self.go_to_pos(user)

    def action_3_down(self, user):  # user go right
        user.freeCount = 0
        user.posy = user.posy + 1
        if user.posy >= self.userMatch.gameinfo.height:
            user.status = constants.Status.STATUS_ELIMINATED_WENT_OUT_MAP.value
            user.lastAction = 6  # eliminated
        else:
            self.go_to_pos(user)

    def action_4_free(self, user):  # user free
        user.freeCount += 1
        if user.freeCount == 1:
            user.energy += int(self.E / 4)
        elif user.freeCount == 2:
            user.energy += int(self.E / 3)
        elif user.freeCount == 3:
            user.energy += int(self.E / 2)
        else:
            user.energy = self.E
        if user.energy > self.E:
            user.energy = self.E

    def action_5_craft(self):
        craftCount = len(self.craftUsers)
        # print ("craftCount",craftCount)
        if craftCount > 0:
            for user in self.craftUsers:
                x = user.posx
                y = user.posy
                key = str(user.posx) + "_" + str(user.posy)
                c = self.craftMap[key]
                m = min(math.ceil(self.map[y][x] / c), 50)
                user.score += m
                # print ("user", user.playerId, m)
            for user in self.craftUsers:
                x = user.posx
                y = user.posy
                key = str(user.posx) + "_" + str(user.posy)
                if key in self.craftMap:
                    c = self.craftMap[key]
                    del self.craftMap[key]
                    m = min(math.ceil(self.map[y][x] / c), 50)
                    self.map[y][x] -= m * c
                    if self.map[y][x] < 0:
                        self.map[y][x] = 0
                        self.energyOnMap[y][x] = ObstacleInfo.types[0]
                    for g in self.stepState.golds:
                        if g.posx == x and g.posy == y:
                            g.amount = self.map[y][x]
                            if g.amount == 0:
                                self.stepState.golds.remove(g)
                                self.add_changed_obstacle(x, y, 0, ObstacleInfo.types[0])
                                if len(self.stepState.golds) == 0:
                                    for player in self.stepState.players:
                                        player.status = constants.Status.STATUS_STOP_EMPTY_GOLD.value
                            break
            self.craftMap = {}

    def invalidAction(self, user):
        user.status = constants.Status.STATUS_ELIMINATED_INVALID_ACTION.value
        user.lastAction = 6  # eliminated

    def go_to_pos(self, user):  # player move to cell(x,y)
        if self.map[user.posy][user.posx] == -1:
            user.energy -= randrange(16) + 5
        elif self.map[user.posy][user.posx] == 0:
            user.energy += self.energyOnMap[user.posy][user.posx]
        elif self.map[user.posy][user.posx] == -2:
            user.energy += self.energyOnMap[user.posy][user.posx]
            self.add_changed_obstacle(user.posx, user.posy, 0, ObstacleInfo.types[0])
        elif self.map[user.posy][user.posx] == -3:
            user.energy += self.energyOnMap[user.posy][user.posx]
            self.add_changed_obstacle(user.posx, user.posy, 3,
                                      self.bog_energy_chain[self.energyOnMap[user.posy][user.posx]])
        else:
            user.energy -= 4
        if user.energy <= 0:
            user.status = constants.Status.STATUS_ELIMINATED_OUT_OF_ENERGY.value
            user.lastAction = 6  # eliminated

    def add_changed_obstacle(self, x, y, t, v):
        added = False
        for o in self.stepState.changedObstacles:
            if o["posx"] == x and o["posy"] == y:
                added = True
                break
        if not added:
            o = {}
            o["posx"] = x
            o["posy"] = y
            o["type"] = t
            o["value"] = v
            self.stepState.changedObstacles.append(o)

    def close(self):
        print("Close socket.")
