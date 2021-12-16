from tqdm import tqdm

import numpy as np
from game import Simulation,  Game, Dirs

class MCTS_Mod:
    #This class inputs a U that gives both 
    #1. a probability distribution p
    #2. a value
    def __init__(self, mdp, N, Q, d, m, U, P):
        self.mdp = mdp
        if N is not None:
            self.N = N
        else:
            self.N = {}
        if Q is not None:
            self.Q = Q
        else:
            self.Q = {}
        if P is not None:
            self.P = P
        else:
            self.P = {}
        self.d = d
        self.m = m
        self.U = U



        self.c = 100
        self.gamma = 0.9


    def TR(self,s, a):
        simulation = Simulation(s)
        simulation.simulate_next_state(a)
 
        return simulation.board
    

    def pi(self, s):
        for k in range(self.m):
            self.simulate(s)

        moves = s.get_available_moves()
        moves_dict = dict({a: self.P[s][a.value]*self.Q[(s,a)] for a in moves})
        dir = max(
             moves_dict, key=moves_dict.get
        )
        return dir

    def explore(self, s):
        moves = s.get_available_moves()
        Ns = sum(self.N[(s, a)] for a in moves)
        
        Qs_ucbs = {a:self.Q[(s,a)] + self.P[s][a.value]*self.c*np.sqrt(Ns/(self.N[(s, a)]+1)) for a in moves}
        dir = max(Qs_ucbs, key=Qs_ucbs.get)
        return dir


    def simulate(self, s, d=None):
        if d is None:
            d = self.d
        if d == 0:
            p, s_val = self.U(s)
            return s_val
        
        moves = s.get_available_moves()
        p, s_val = self.U(s)
        s_val = s_val[0]
        p = p/sum(p)
        if len(moves) == 0:
            return s_val
        if (s, moves[0]) not in self.N:
            for a in Dirs:
                self.N[(s, a)] = 0
                self.Q[(s, a)] = 0
            
            self.P[s] = p
            return s_val
        a = self.explore(s)
        s_prime = self.TR(s, a)
        q = self.gamma*self.simulate(s_prime, d-1)
        self.N[(s, a)] += 1
        self.Q[(s, a)] += (q-self.Q[(s, a)])/self.N[(s, a)]
        return q




