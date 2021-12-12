import numpy as np
from enum import Enum
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
        start_idx = (np.random.randint(0, self.dims[0]),
                    np.random.randint(0, self.dims[1]))
        rand_val = np.random.randint(1, 2+1)

        self.board[start_idx] = rand_val

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

    def move(self, dir):

        if dir == Dirs.LEFT:
            for row in self.board:
                for _ in range(self.dims[1]):
                    self.move_row_left(row)

        if dir == Dirs.RIGHT:
            for row in self.board:
                for _ in range(self.dims[1]):
                    self.move_row_right(row)

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



if __name__ == '__main__':
    board = Board()

    board.display()

    board.move(Dirs.LEFT)
    board.display()


        
        
