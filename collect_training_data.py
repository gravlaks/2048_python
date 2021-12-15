from modified_mcts import MCTS_Mod
import torch
from game import Game
import numpy as np
from neural_network import Net
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
def play_one_game(network):
    game = Game()
    mcts = MCTS_Mod(
        mdp = None, 
        N=None,
        Q = None,
        d = 2,
        m = 5,
        U = U_numpy(network),
        P = None
    )
    moves = game.board.get_available_moves()
    game.board.display()
    while moves:
        s = game.board
        dir = mcts.pi(s)
        
        game.take_turn(dir)
        moves = game.board.get_available_moves()
        s.display()
    return game.board.get_value()


if __name__ == '__main__':
    net = Net()
    net = net.float()
    net.eval()
    play_one_game(net)
