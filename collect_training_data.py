from modified_mcts import MCTS_Mod
import torch
from game import Game, Simulation
import numpy as np
from neural_network import Net, train_network
import pickle
from datetime import datetime
gameCount = 10
counter = 0
def U_numpy(network):
    def U(s):
        global counter
        counter += 1
        s = torch.from_numpy(s.board).float()
        s = s.expand(1, 1, -1, -1)
        out = network(s)
        
        return (out[0, :4].detach().numpy().flatten(), 
                out[0, 4:5].detach().numpy().flatten())
    return U

def random_rollout(s):
    simulation = Simulation(s)
    t1 = datetime.now()
    ret = (np.ones((4, ))*1/4, simulation.random_rollout())
    #print(datetime.now()-t1)
    return ret
    


def play_one_game(mcts):
    game = Game()
   
    moves = game.board.get_available_moves()

    boards = []
    labels = []
    while moves:
        s = game.board
        board_copy = s.board.copy()
        boards.append(board_copy)
        dir = mcts.pi(s)
        arr = np.zeros((4,))
        arr[dir.value] = 1
        labels.append(arr)
        
        game.take_turn(dir)
        moves = s.get_available_moves()
    s.display()
    exp_val = game.board.get_value()
    for i, label in enumerate(labels):
        labels[i] = np.hstack((label, exp_val))
    return boards, labels

def collect(episode_count, mcts):
    data = []
    target = []
    
    for episode in range(episode_count):
        boards, labels = play_one_game(mcts)
        data = data+boards
        target = target+labels
        print("episode done")
    return np.array(data[1:]), np.array(target[1:])

def save(data, target, filename):
    np.save(f"datasets/data_{filename}.np", data)
    with open(f"datasets/data_{filename}.pkl", "wb") as f:
        pickle.dump(target, f)
if __name__ == '__main__':
    net = Net()
    net = net.float()
    net.eval()
    mcts = MCTS_Mod(
        mdp = None, 
        N=None,
        Q = None,
        d = 3,
        m = 50,
        U = U_numpy(net),
        P = None
    )

    for i in range(20):
        data, target = collect(episode_count=20, mcts=mcts)
        #save(data, target, f"iteration_{i}")
        train_network(net, data, target, epochs_count=500)
    data, target = collect(episode_count=30, mcts=mcts)

  