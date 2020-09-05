from MINER_STATE import State
import numpy as np
from Graph import Graph
from random import randrange

from Functions import valid, softmax
import Constants

class PlayerInfo:
    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = 0
        self.freeCount = 0 
        

class G_BOT5:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, id,estWood=-1,pEnergyToStep=-1,pStepToGold=-1, strategy = -1):
        self.state = State()
        self.info = PlayerInfo(id)

        if (estWood==-1): #random strenght 
            estWood         = (5 + randrange(16))
            pEnergyToStep   = (2 + randrange(9))* 5
            pStepToGold     = (1 + randrange(6) )*50

        self.estWood = estWood
        self.pEnergyToStep = pEnergyToStep
        self.pStepToGold  = pStepToGold
        #print ("AddG_BOT",estWood,pEnergyToStep,pStepToGold)
        self.tx = -1
        self.ty = -1

        if (strategy == -1):
            strategy = np.random.choice(range(6))-1

        self.selectTargetOption = strategy
        #print ("strategy",strategy)

    def next_action(self):

        #if (self.info.status!=0 and self.state.stepCount < 100):
        #    print ("WTF",self.info.status)

        countPlayerAtGoldMine = 0
        x, y= self.info.posx,self.info.posy
        r_Action = self.ACTION_FREE #for safe

        if (self.isKeepFree ):
            self.isKeepFree = False
            return r_Action
        
        # 1st rule. Heighest Priority. Craft & Survive 
        if (valid(y,x)):
            
            goldOnGround =  self.state.mapInfo.gold_amount(x, y)
            countPlayerAtGoldMine = 1
            
            for player in self.state.players:
                px,py = player['posx'],player['posy']
                if (px==x and py==y):
                    countPlayerAtGoldMine+= 1
            
            if ( goldOnGround > 0 and countPlayerAtGoldMine > 0 ):
                if ( goldOnGround/countPlayerAtGoldMine > 0 and self.info.energy > 5):
                    r_Action = self.ACTION_CRAFT
                    self.tx = -1
            else :
                g = Graph(Constants.MAP_MAX_Y,Constants.MAP_MAX_X)
                g.convertToMap(state = self.state, estWood =self.estWood, botInfo = self.info , isBot = True)
                g.BFS()
                
                if (self.tx == -1 or self.state.mapInfo.gold_amount(self.ty,self.tx)==0 ):
                    #print ("Change Target")
                    self.tx,self.ty = g.getRandomTarget(y,x,self.selectTargetOption)
                
                #print ("Target",self.tx,self.ty)
                ny,nx = g.traceBack(self.tx,self.ty)
                ny,nx = int(ny), int (nx)

                if (ny ==- 1):
                    self.tx =-1
                    return self.ACTION_FREE
                
                typeOb = self.state.mapInfo.get_obstacle(nx,ny)
                
                nextTrap = g.boardMap[ny,nx]
                if (typeOb ==  1 ):    # WOOOD
                    nextTrap = 20

                if (  nextTrap >= self.info.energy  ):
                    r_Action = self.ACTION_FREE
                else:
                    if (ny == y):
                        if (nx > x):
                            r_Action=  self.ACTION_GO_RIGHT
                        elif (nx<x):
                            r_Action=  self.ACTION_GO_LEFT
                    else: #nx==x
                        if (ny > y):
                            r_Action=  self.ACTION_GO_DOWN
                        elif (ny < y):
                            r_Action=  self.ACTION_GO_UP

        else :
            print ("BOT 5 INVALID WTF")

        if (r_Action < 4 and self.info.energy <= 13  and self.state.stepCount < 90):
            self.isKeepFree = True
        return r_Action
    
    def new_game(self, data):
        try:
            self.isKeepFree = False
            self.state.init_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def new_state(self, data):
        # action = self.next_action();
        # self.socket.send(action)
        try:
            self.state.update_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def printInfo(self):
        print ("G_BOT",self.info.playerId,self.estWood,self.pEnergyToStep,self.pStepToGold,self.info.score,self.info.energy)
