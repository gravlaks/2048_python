from board import Board, Dirs
import numpy as np
class Game:
    def __init__(self, board=None, dims=(4, 4), policy_function=None):
        self.dims = dims
        
        if policy_function is not None:
            self.policy_function = policy_function
        else:
            self.policy_function = self.random_move
        
        if board is None:
            self.reset()
        else:
            self.board = Board(board.board)
    def take_turn(self, dir):
        self.board.take_turn(dir)
    def move(self, dir):
        self.board.move(dir)

    def reset(self):
        self.board = Board()

    def computer_move(self):
        move = self.policy_function(self.board)
        self.board.take_turn(move)

    def ask_for_move(self):
        moves = self.board.get_available_moves()
        print("Choose a move:")

        for i, move in enumerate(Dirs):
            if move in moves:
                print(i, ": ",move.name)
        try:
            move = Dirs(int(input(">")))
        except:
            move = None
        while move not in moves:
            try:
                move = Dirs(int(input("Not valid, try again\n>")))
            except Exception as e:
                continue
        return move
    def human_move(self):
        move = self.ask_for_move()
        self.board.take_turn(move)
    def random_move(self, board):
        moves = board.get_available_moves()
        return moves[np.random.randint(0, len(moves))]
    

    def play_human(self):
        while not self.board.full():
            self.board.display()
            self.human_move()

class Simulation:  

    def __init__(self, board):
        self.board = Board(board.board.copy())
    
    def simulate_next_state(self, a):
        self.board.take_turn(a)
        return self.board
    def random_rollout(self):
        while not self.board.full():

            moves = self.board.get_available_moves()
            self.board.take_turn(moves[np.random.randint(0, len(moves))])
        return self.board.get_value()

if __name__ == '__main__':
    game = Game()
    game.play_human()