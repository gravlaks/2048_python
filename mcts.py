import numpy as np
from game import Game, Simulation

from tqdm import tqdm

class MCTS:
    def __init__(self, mdp, N, Q, d, m, U):
        self.mdp = mdp
        if N is not None:
            self.N = N
        else:
            self.N = {}
        if Q is not None:
            self.Q = Q
        else:
            self.Q = {}
        self.d = d
        self.m = m
        self.U = U



        self.c = 0.1
        self.gamma = 0.9


    def TR(self,s, a):
        simulation = Simulation(s)
        simulation.simulate_next_state(a)
 
        return simulation.board
    

    def pi(self, s):
        for k in tqdm(range(self.m)):
            self.simulate(s)

        moves = s.get_available_moves()
        
        moves_dict = dict({a: self.Q[(s,a)] for a in moves})
        dir = max(
             moves_dict, key=moves_dict.get
        )
        return dir

    def explore(self, s):
        moves = s.get_available_moves()
        Ns = sum(self.N[(s, a)] for a in moves)
        eps = 1e-6
        Ns = max(1, Ns)

        
        Qs_ucbs = {a:self.Q[(s,a)] + self.c*np.sqrt(np.log(Ns)/max(self.N[(s,a)], eps)) for a in moves}
        dir = max(Qs_ucbs, key=Qs_ucbs.get)
        return dir


    def simulate(self, s, d=None):
        if d is None:
            d = self.d
        if d == 0:
            return self.U(s)
        
        moves = s.get_available_moves()
        if len(moves) == 0:
            return s.get_value()
        if (s, moves[0]) not in self.N:
            for a in moves:
                self.N[(s, a)] = 0
                self.Q[(s, a)] = 0
            return self.U(s)
        a = self.explore(s)
        s_prime = self.TR(s, a)
        q = self.gamma*self.simulate(s_prime, d-1)
        self.N[(s, a)] += 1
        self.Q[(s, a)] += (q-self.Q[(s, a)])/self.N[(s, a)]
        return q

def U(s):
    simulation = Simulation(s)
    return simulation.random_rollout()

if __name__ == '__main__':
    game = Game()
    mcts = MCTS(
        mdp = None, 
        N=None,
        Q = None,
        d = 2,
        m = 5,
        U = U
    )
    moves = game.board.get_available_moves()
    game.board.display()
    while moves:
        s = game.board
        dir = mcts.pi(s)
        
        game.take_turn(dir)
        moves = game.board.get_available_moves()
        s.display()
    print(s.get_exp_value())

