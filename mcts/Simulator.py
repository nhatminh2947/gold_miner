import numpy as np

import constants


class MinerSimulator:
    def __init__(self, map, positions, energies, score):
        self._map = np.asarray(map)  # store enery and gold
        self._positions = positions
        self._energies = energies
        self._score = score
        self._free_count = [0, 0, 0, 0]

        self.dx = [-1, 1, 0, 1, 0, 0]  # LEFT, RIGHT, UP, DOWN, FREE, CRAFT
        self.dy = [0, 0, -1, 1, 0, 0]

    def is_eliminated(self, id):
        return (self._energies[id] <= 0) or not self.inside(id)

    def inside(self, id):
        return (0 <= self._positions[id][0] < 9) and (0 <= self._positions[id][1] < 21)

    def step(self, actions):
        print(actions)
        crafting = {}
        move_in_pos = set()

        for id, action in enumerate(actions):
            if self.is_eliminated(id):
                continue

            self._positions[id] = (self._positions[id][0] + self.dy[action], self._positions[id][1] + self.dx[action])

            if self.is_eliminated(id):
                continue

            self.handle_energy_loss(id, action, self._positions[id])

            if action == constants.Action.ACTION_CRAFT.value:
                if str(self._positions[id]) not in crafting:
                    crafting[str(self._positions[id])] = []

                crafting[str(self._positions[id])].append(id)
            elif action != constants.Action.ACTION_FREE.value:
                move_in_pos.add(self._positions[id])

        self.handle_move_in(move_in_pos)

        return self._map, self._positions, self._energies, self._score

    def handle_move_in(self, move_in_pos):
        for pos in move_in_pos:
            if self._map[pos] == -10:
                self._map[pos] = -1
            elif self._map[pos] == -5:
                self._map[pos] = -20
            elif self._map[pos] == -20:
                self._map[pos] = -40
            elif self._map[pos] == -40:
                self._map[pos] = -100

    def handle_energy_loss(self, id, action, pos):
        if action < 4:  # Move in
            if self._map[pos] == 0:  # Forest
                self._energies[id] += -np.random.randint(5, 21)
            elif self._map[pos] > 0:  # Gold
                self._energies[id] += -4
            else:
                self._energies[id] += self._map[pos]

            self._free_count[id] = 0
        elif action == 4:  # FREE
            self._energies[id] = max(
                50,
                self._energies[id] + 12
                + (4 if self._free_count[id] > 0 else 0)
                + (9 if self._free_count[id] > 1 else 0)
            )

            self._free_count[id] += 1
        else:  # CRAFT
            self._energies[id] += -(5 if self._map[pos] > 0 else 10)
            self._free_count[id] = 0

    def handle_craft(self, crafting):
        for pos in crafting:
            if self._map[pos] > 0:
                for id in crafting[pos]:
                    self._score[id] += min(50, self._map[pos] / len(crafting[pos]))

                self._map[pos] -= min(50 * len(crafting[pos]), self._map[pos])

                if self._map[pos] == 0:
                    self._map[pos] = -1


if __name__ == '__main__':
    import utils
    import json

    map_0 = json.loads(
        "[[0,0,-2,100,0,0,-1,-1,-3,0,0,0,-1,-1,0,0,-3,0,-1,-1,0],[-1,-1,-2,0,0,0,-3,-1,0,-2,0,0,0,-1,0,-1,0,-2,-1,0,0],[0,0,-1,0,0,0,0,-1,-1,-1,0,0,100,0,0,0,0,50,-2,0,0],[0,0,0,0,-2,0,0,0,0,0,0,0,-1,50,-2,0,0,-1,-1,0,0],[-2,0,200,-2,-2,300,0,0,-2,-2,0,0,-3,0,-1,0,0,-3,-1,0,0],[0,-1,0,0,0,0,0,-3,0,0,-1,-1,0,0,0,0,0,0,-2,0,0],[0,-1,-1,0,0,-1,-1,0,0,700,-1,0,0,0,-2,-1,-1,0,0,0,100],[0,0,0,500,0,0,-1,0,-2,-2,-1,-1,0,0,-2,0,-3,0,0,-1,0],[-1,-1,0,-2,0,-1,-2,0,400,-2,-1,-1,500,0,-2,0,-3,100,0,0,0]]")
    position = [(0, 0), (0, 0), (0, 0), (0, 0)]
    energy = [50, 50, 50, 50]
    score = [0, 0, 0, 0]
    map_value = np.zeros((9, 21), dtype=int)
    for i in range(9):
        for j in range(21):
            if map_0[i][j] == 0:
                map_value[i][j] = -1
            elif map_0[i][j] == -1:
                map_value[i][j] = 0
            elif map_0[i][j] == -2:
                map_value[i][j] = -10
            elif map_0[i][j] == -3:
                map_value[i][j] = -5
            else:
                map_value[i][j] = map_0[i][j]

    sim = MinerSimulator(map_value, position, energy, score)

    utils.print_map_simulator(map_value, position, energy, score)
    map_value, position, energy, score = sim.step([0, 1, 1, 1])
    utils.print_map_simulator(map_value, position, energy, score)
    map_value, position, energy, score = sim.step([0, 1, 1, 1])
    utils.print_map_simulator(map_value, position, energy, score)
    map_value, position, energy, score = sim.step([0, 1, 1, 1])
    utils.print_map_simulator(map_value, position, energy, score)
    map_value, position, energy, score = sim.step([0, 1, 1, 1])
    utils.print_map_simulator(map_value, position, energy, score)
