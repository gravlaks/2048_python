from board import Board, Dirs
import numpy as np
def test_merge_left():

    board = Board()
    board.board = np.array([
       [0, 0, 2, 2],
       [0, 0, 0, 0],
        [0, 0, 0, 0], 
        [0, 0, 0,0] 
    ])
    board.move(Dirs.LEFT)


    assert(
        np.allclose(board.board, 
        np.array([
       [3, 0, 0, 0],
       [0, 0, 0, 0],
        [0, 0, 0, 0], 
        [0, 0, 0,0] 
    ]))
    ), "Easy merge failed"

    board.board = np.array([
       [0, 1, 2, 2],
       [0, 0, 0, 0],
        [0, 0, 0, 0], 
        [0, 0, 0,0] 
    ])

    
    board.move(Dirs.LEFT)

    assert(
        np.allclose(board.board, 
        np.array([
       [1, 3, 0, 0],
       [0, 0, 0, 0],
        [0, 0, 0, 0], 
        [0, 0, 0,0] 
    ]))
    ), "Second merge failed"

    board.board = np.array([
       [2, 1, 0, 2],
       [0, 4, 0, 4],
        [0, 3, 0, 8], 
        [0, 0, 1,0] 
    ])
    board.move(Dirs.LEFT)
    assert(
        np.allclose(board.board, 
        np.array([
       [2, 1, 2, 0],
       [5, 0, 0, 0],
        [3, 8, 0, 0], 
        [1, 0, 0,0] 
    ]))
    ), "Third merge failed"

    
def test_move_right():
    board = Board()
    board.board = np.array([
       [0, 0, 2, 2],
       [0, 3, 0, 3],
        [1, 0, 4, 1], 
        [10, 10, 0,10] 
    ])
    board.move(Dirs.RIGHT)

    assert(
        np.allclose(board.board, 
        np.array([
       [0, 0, 0, 3],
       [0, 0, 0, 4],
        [0, 1, 4, 1], 
        [0, 0, 10,11] 
    ]))
    ), "Easy merge failed"

def test_move_down():
    board = Board()
    board.board = np.array([
       [10, 1, 2, 4],
       [1, 3, 0, 4],
        [1, 3, 3, 1], 
        [10, 0, 0,10] 
    ])
    board.move(Dirs.DOWN)

    assert(
        np.allclose(board.board, 
        np.array([
       [0, 0, 0, 0],
       [10, 0, 0, 5],
        [2, 1, 2, 1], 
        [10, 4, 3,10] 
    ]))
    ), "Easy merge failed"

def test_move_up():
    board = Board()
    board.board = np.array([
       [10, 1, 2, 4],
       [1, 3, 0, 4],
        [1, 3, 3, 1], 
        [10, 0, 0,10] 
    ])
    board.move(Dirs.UP)
    assert(
        np.allclose(board.board, 
        np.array([
       [10, 1, 2, 5],
       [2, 4, 3, 1],
        [10, 0, 0, 10], 
        [0, 0, 0,0] 
    ]))
    ), "Easy merge failed"

def test_get_available_moves():

    board = Board()
    board.board = np.array([
       [10, 1, 2, 4],
       [1, 3, 0, 4],
        [1, 3, 3, 1], 
        [10, 0, 0,10] 
    ])  
    assert(
        board.get_available_moves() == [Dirs.LEFT, Dirs.RIGHT, Dirs.UP, Dirs.DOWN]
    ), "easy"
    board.board = np.array([
       [10, 1, 2, 2],
       [1, 3, 1, 4],
        [2, 8, 3, 1], 
        [10, 4, 1,10] 
    ])  
    assert(
        board.get_available_moves() == [Dirs.LEFT, Dirs.RIGHT]
    ), "easy"

    assert(np.allclose(np.array([
       [10, 1, 2, 2],
       [1, 3, 1, 4],
        [2, 8, 3, 1], 
        [10, 4, 1,10] 
    ]) , board.board)), "board changed during checking"

    
if __name__ == '__main__':
    test_merge_left()
    test_move_right()
    test_move_up()
    test_move_down()
    test_get_available_moves()