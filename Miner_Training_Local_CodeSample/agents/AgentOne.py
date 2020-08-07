from constants import Action
from . import BaseAgent


class AgentOne(BaseAgent):
    def __init__(self, id):
        super().__init__(id)

    def next_action(self):
        if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
            if self.info.energy >= 6:
                return Action.ACTION_CRAFT.value
            else:
                return Action.ACTION_FREE.value
        if self.info.energy < 5:
            return Action.ACTION_FREE.value
        else:
            action = Action.ACTION_GO_UP.value
            if self.info.posy % 2 == 0:
                if self.info.posx < self.state.mapInfo.max_x:
                    action = Action.ACTION_GO_RIGHT.value
            else:
                if self.info.posx > 0:
                    action = Action.ACTION_GO_LEFT.value
                else:
                    action = Action.ACTION_GO_DOWN.value
            return action
