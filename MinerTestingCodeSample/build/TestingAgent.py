from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

from MinerEnv import MinerEnv
from fifth_model import FifthModel
import sys
import torch

ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5
i = 1
HOST = "localhost"
PORT = 1110 + i
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

model = FifthModel()
model.load_state_dict(
    torch.load(f"./TrainedModels/model_{i}.pt"))
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
    obs = minerEnv.get_state()  ##Getting an initial state
    while not minerEnv.check_terminate():
        try:
            action = model.predict(obs)  # Getting an action from the trained model
            print("next action = ", action)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            s_next = minerEnv.get_state()  # Getting a new state
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
