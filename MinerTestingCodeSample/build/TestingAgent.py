from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
from collections import deque
from MinerEnv import MinerEnv
from fifth_model import FifthModel
from seventh_model import SeventhModel
import sys
import torch
import utils

ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5

model_version = SeventhModel

id = 0
HOST = "localhost"
PORT = 1110 + id
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

models = [model_version(), model_version(), model_version(), model_version(),
          model_version(), model_version(), model_version(), model_version()]

for i, model in enumerate(models):
    model.load_state_dict(torch.load(f"./TrainedModels/model_{i}.pt"))
    model.to('cpu')

# Choosing a map in the list
# mapID = np.random.randint(1, 6)  # Choosing a map ID from 5 maps in Maps folder randomly
# posID_x = np.random.randint(21)  # Choosing a initial position of the DQN agent on X-axes randomly
# posID_y = np.random.randint(9)  # Choosing a initial position of the DQN agent on Y-axes randomly
# Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
# request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
# Send the request to the game environment (GAME_SOCKET_DUMMY.py)

try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game

    # minerEnv.send_map_info(request)
    minerEnv.reset()
    last_3_actions = deque([4, 4, 4], maxlen=3)
    obs, raw_obs = minerEnv.get_state_v2(last_3_actions)  ##Getting an initial state

    while not minerEnv.check_terminate():
        try:
            utils.print_map(raw_obs)
            votes = {i: 0 for i in range(6)}
            best_model_act = models[7].predict(obs)
            # votes[best_model_act] += 1
            for model in models:
                votes[model.predict(obs)] += 1

            choosen_move = None
            max_count = -1
            for move in votes:
                if votes[move] > max_count:
                    choosen_move = move
                    max_count = votes[move]

            if votes[best_model_act] == max_count:
                choosen_move = best_model_act

            action = choosen_move  # Getting an action from the trained model
            print("next action = ", action)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            last_3_actions.append(best_model_act)
            s_next, raw_obs = minerEnv.get_state_v2(last_3_actions)  # Getting a new state
            obs = s_next

        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Finished.")
            break

    print(f"Final score: {minerEnv.state.score}")
except Exception as e:
    import traceback

    traceback.print_exc()
print("End game.")
