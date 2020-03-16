import matplotlib.pyplot as plt
import tkinter as tk
from .game import Board, Blokus, Move
from .pieces import PIECES


### TODO define some standardized representation of the board + pieces left
### encoded in JSON or Protobuf, or w/e

### Then build this board_ui (a client) to read the board state and display it

# base_bg_color = "#eee4da"
player_colors = ["#ffffff", "#000000"]

class BlokusGameBoard(tk.Frame):
    def __init__(self, board, master=None):
        super().__init__(master)
        self.board = board
        self.height = board.grid.shape[2]
        self.width = board.grid.shape[3]
        self.init_squares = board.init_squares
        self.num_players = board.num_players
        self.master = master
        print(f"{self.num_players}-player board, size {self.height}x{self.width}, init_squares={self.init_squares}")

        for r in range(self.height):
            self.rowconfigure(r, minsize=20)
        for c in range(self.width):
            self.columnconfigure(c, minsize=20)

        self.tiles = []
        for r in range(self.height):
            self.tiles.append([])
            for c in range(self.width):
                # the board includes padding
                if c == 0 or c == self.width - 1 or r == 0 or r == self.height - 1:
                    self.tiles[r].append(tk.Label(
                        self, anchor=tk.CENTER
                    ))
                    # add coordinate grid reference
                    if (c == 0 or c == self.width - 1) and (r != 0 and r != self.height - 1):
                        self.tiles[r][c]['text'] = str(r)
                    elif (r == self.height - 1 or r == 0) and (c != 0 and c != self.width - 1):
                        self.tiles[r][c]['text'] = str(chr(96 + c))
                else:
                    self.tiles[r].append(tk.Label(
                        self, anchor=tk.CENTER,
                        # bg=base_bg_color,
                        bd=1, relief=tk.SOLID
                    ))
                self.tiles[r][c].grid(row=r, column=c, sticky=tk.N+tk.S+tk.W+tk.E)
        for init_r, init_c in self.init_squares:
            self.tiles[init_r][init_c]['text'] = 'O'
        self.grid()

    def draw_tiles(self, game):
        for player in range(self.num_players):
            for r in range(self.height):
                for c in range(self.width):
                    if self.board.grid[player, 0, r, c] == 1:
                        self.tiles[r][c]['bg'] = player_colors[player]
                        self.tiles[r][c].config(relief=tk.FLAT)

class BlokusGameWindow(tk.Frame):
    def __init__(self, game=Blokus(), master=None):
        super().__init__(master)
        self.master = master
        self.game = game
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        self.game_tiles = BlokusGameBoard(self.game.board, master=self)
        self.game_tiles.draw_tiles(self.game)
        self.game_tiles.grid()

        self.scores_strvar = tk.StringVar()
        self.game_scores = tk.Label(self, textvariable=self.scores_strvar)
        self.draw_scores()
        self.game_scores.grid()

    def draw_scores(self):
        scores_str = ""
        for i in range(len(self.game.scores)):
            scores_str += "Player " + str(i + 1) + " score: " + str(self.game.scores[i]) + "\n"
        self.scores_strvar.set(scores_str)


def test_board_class():
    # blokus duo is 14x14 but this board dimension includes the 1-width padding
    # around the board for collision checks
    test_board = Board(2, 16, 16, [(5, 5), (10, 10)])

    test_board.place(0, PIECES["N"], 0, 4, 4)
    test_board.show()
    plt.show()

    test_board.place(1, PIECES["X"], 0, 8, 8)
    test_board.show()
    plt.show()


def main():
    root = tk.Tk()
    root.geometry("800x800")

    test_game = Blokus()
    move1 = Move(0, "X", 0, 3, 3)
    test_game.place(move1)
    move2 = Move(1, "Z5", 0, 7, 7)
    test_game.place(move2)

    app = BlokusGameWindow(game=test_game, master=root)
    app.mainloop()

if __name__ == "__main__":
    main()
