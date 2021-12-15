from modified_mcts import MCTS_Mod
import torch
from game import Game
import numpy as np
from neural_network import Net
from sklearn.preprocessing import OneHotEncoder
gameCount = 10

def U_numpy(network):
    print("u numpy")
    def U(s):
        s = torch.from_numpy(s.board).float()
        s = s.expand(1, 1, -1, -1)
        print(s.size())
        out = network(s)
        print("out",type(out))
        return (out[0].detach().numpy().flatten(), 
                out[1].detach().numpy().flatten())
    return U
def play_one_game(mcts):
    game = Game()
   
    moves = game.board.get_available_moves()
    game.board.display()

    dims = (1,4, 4)
    boards = []
    labels = []
    while moves:
        s = game.board
        board_copy = s.board.copy()
        boards.append(board_copy)
        dir = mcts.pi(s)
        arr = np.zeros((4,))
        arr[dir.value] = 1
        labels.append([arr])
        
        game.take_turn(dir)
        moves = s.get_available_moves()
        s.display()
    print(game.board.get_exp_value())
    exp_val = game.board.get_exp_value()
    for label in labels:
        label.append(exp_val)
    return boards, labels

def collect(network, episode_count, mcts):
    data = []
    target = []
    
    for episode in range(episode_count):
        boards, labels = play_one_game(mcts)
        data = data+ boards
        target = target+labels
    print(target[1])
    return data[1:], target[1:]
if __name__ == '__main__':
    net = Net()
    net = net.float()
    net.eval()
    mcts = MCTS_Mod(
        mdp = None, 
        N=None,
        Q = None,
        d = 2,
        m = 2,
        U = U_numpy(net),
        P = None
    )
    data, target = collect(net,2, mcts)
    data = np.array(data)
    print(target)
    print(target[0])
