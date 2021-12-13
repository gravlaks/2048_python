import numpy as np
from enum import Enum

from numpy.random.mtrand import rand
class Dirs(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class Board:
    def __init__(self, dims = (4, 4)):
        self.dims = dims
        self.reset()
    def reset(self):
        self.board = np.zeros(self.dims)
        self.add_new_num()
        self.add_new_num()

    def display(self):
        powered = 2**self.board
        print()
        for row in powered:
            for elem in row:
                if (elem == 1):
                    print(str(0).ljust(5), end=" ")
                else:
                    print(str(int(elem)).ljust(5), end =" ")
            print()
        print()
    def take_turn(self, dir):
        assert(dir in self.get_available_moves()), "cannot take that move"
        self.move(dir)
        self.add_new_num()

    def full(self):
        return len(self.get_available_moves())==0

    def move(self, dir, copy = False):
        if copy:
            board = self.board.copy()
        else:
            board = self.board
        if dir == Dirs.LEFT:
            for row in board:
                for _ in range(self.dims[1]):
                    self.move_row_left(row)

        if dir == Dirs.RIGHT:
            for row in board:
                for _ in range(self.dims[1]):
                    self.move_row_right(row)
        
        if dir == Dirs.UP:
            for j in range(self.dims[1]):
                for _ in range(self.dims[0]):
                    self.move_row_left(board[:, j])

        if dir == Dirs.DOWN:
            for j in range(self.dims[1]):
                for _ in range(self.dims[0]):
                    self.move_row_right(board[:, j])

       
        
        if copy:
            return board
    def add_new_num(self):
        l_idxs, r_idxs = np.nonzero(self.board==0)
        rand_idx = np.random.randint(0, len(l_idxs))
        print(l_idxs[rand_idx], r_idxs[rand_idx])
        self.board[l_idxs[rand_idx], r_idxs[rand_idx]] = np.random.randint(1,3)


    def move_row_left(self, row):
        if len(row) == 1:
            return
        if row[0] == 0:
            row[0] = row[1]
            row[1] = 0
        elif row[0] == row[1]:
            row[0] = row[0]+1
            row[1] = 0
        self.move_row_left(row[1:])

    def move_row_right(self, row):
        if len(row) == 1:
            return
        if row[-1] == 0:
            row[-1] = row[-2]
            row[-2] = 0
        elif row[-1] == row[-2]:
            row[-1] = row[-2]+1
            row[-2] = 0
        
        self.move_row_right(row[:-1])

    def get_available_moves(self):
        moves = []
        board = self.board
        for dir in Dirs:
            potential = self.move(dir, copy=True)
            if not np.allclose(potential, board):
                moves.append(dir)
        return moves
if __name__ == '__main__':
    board = Board()

    board.display()

    board.take_turn(Dirs.LEFT)
    board.display()


        
        
