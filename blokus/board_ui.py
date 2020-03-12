import matplotlib.pyplot as plt
from .game import Board
from .pieces import PIECES

def main():
    # blokus duo is 14x14 but this board dimension includes the 1-width padding
    # around the board for collision checks
    test_board = Board(2, 16, 16, [(5, 5), (10, 10)])

    test_board.place(0, PIECES["N"], 0, 4, 4)
    test_board.show()
    plt.show()

    test_board.place(1, PIECES["X"], 0, 8, 8)
    test_board.show()
    plt.show()


if __name__ == "__main__":
    main()
