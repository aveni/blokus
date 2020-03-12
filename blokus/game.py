import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d
from typing import List, Tuple, Callable
import json
from typing import NamedTuple
import copy
from .profiler import Timer
from .pieces import Piece, PIECES


piece_colors = [(0, 1, 0), (1, 1, 1), (0, 0, 0), (1, 0, 0)]
overview1_colors = [(0, 0, 0), (1, 1, 1), (1, 0, 0)]
overview2_colors = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 0, 1)]
overview4_colors = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 0, 1), (0, 1, 0), "gold"]

piece_cm = LinearSegmentedColormap.from_list("piece", piece_colors, N=4)
overview_cms = {
    1: LinearSegmentedColormap.from_list("overview1", overview1_colors, N=3),
    2: LinearSegmentedColormap.from_list("overview2", overview2_colors, N=4),
    4: LinearSegmentedColormap.from_list("overview4", overview4_colors, N=6),
}




class Board:
    def __init__(self, num_players, height, width, init_squares):
        self.num_players = num_players
        self.init_squares = init_squares
        self.grid = np.zeros((num_players, 3, height, width), dtype="float32")
        for i, (r, c) in enumerate(init_squares):
            self.grid[i, 2, r, c] = 1.0
        self.grid[:, 1, 0, :] = 1.0
        self.grid[:, 1, -1, :] = 1.0
        self.grid[:, 1, :, 0] = 1.0
        self.grid[:, 1, :, -1] = 1.0
        self.cm = overview_cms[self.num_players]

    def place(self, player, piece, orientation, r, c):
        info = piece.orientations[orientation]
        h, w, = info.shape[1], info.shape[2]

        # Place piece into player's piece grid
        self.grid[player, 0, r : r + h, c : c + w] = np.logical_or(
            self.grid[player, 0, r : r + h, c : c + w], info[0]
        )
        # Place dead into player's dead grid
        self.grid[player, 1, r : r + h, c : c + w] = np.logical_or(
            self.grid[player, 1, r : r + h, c : c + w], info[1]
        )
        # Place live into player's live grid
        self.grid[player, 2, r : r + h, c : c + w] = np.logical_or(
            self.grid[player, 2, r : r + h, c : c + w], info[2]
        )
        # Place player's dead into player's live grid
        self.grid[player, 2, r : r + h, c : c + w] = np.logical_and(
            self.grid[player, 2, r : r + h, c : c + w],
            np.logical_not(self.grid[player, 1, r : r + h, c : c + w]),
        )

        # Place piece into all player's dead grids
        self.grid[:, 1, r : r + h, c : c + w] = np.logical_or(
            self.grid[:, 1, r : r + h, c : c + w], info[0]
        )
        # Place piece into all player's live grids
        self.grid[:, 2, r : r + h, c : c + w] = np.logical_and(
            self.grid[:, 2, r : r + h, c : c + w], np.logical_not(info[0])
        )

    def show(self):
        fig, axes = plt.subplots(
            1, self.num_players + 1, figsize=(5 * (self.num_players + 1), 5)
        )

        overview = np.sum(
            [(i + 1) * self.grid[i, 0] for i in range(self.num_players)], axis=0
        )
        overview[0, :] = -1
        overview[-1, :] = -1
        overview[:, 0] = -1
        overview[:, -1] = -1

        axes[0].imshow(overview, cmap=self.cm, vmin=-1, vmax=self.num_players)
        axes[0].set_title("Pieces on board")

        for i in range(self.num_players):
            player_image = self.grid[i, 2] * -1
            player_image += self.grid[i, 1] * 2
            player_image -= self.grid[i, 0]
            axes[i + 1].imshow(player_image, cmap=piece_cm)
            axes[i + 1].set_title(f"Player {i} board view")

    def to_dict(self):
        return {"grid": self.grid.tolist()}

    def from_dict(self, num_players: int):
        raise NotImplementedError


class Move(NamedTuple):
    player: int
    piece: str
    orientation: int
    row: int
    col: int

    def __repr__(self):
        return f"{self.player}, Piece = {self.piece}, Orient = {self.orientation}, Loc = ({self.row}, {self.col})"


class Blokus:
    def __init__(
        self,
        num_players: int = 2,
        board_h: int = 16,
        board_w: int = 16,
        init_squares: List[Tuple[int, int]] = [(5, 5), (10, 10)],
        viz=False,
    ):
        self.board = Board(num_players, board_h, board_w, init_squares)
        self.pieces = [list(PIECES.keys()) for _ in range(num_players)]
        self.num_players = num_players
        self.cur_player = 0
        self.scores = [0] * num_players
        self.viz = viz

        self._available_moves = None

        if self.viz:
            self.show()

    def place(self, move: Move):
        player = move.player
        assert player == self.cur_player
        assert move.piece in self.pieces[player]

        self.board.place(
            player, PIECES[move.piece], move.orientation, move.row, move.col
        )
        self.pieces[player].remove(move.piece)
        self.scores[player] += PIECES[move.piece].tiles

        self.advance()
        while self.available_moves() == [] and self.cur_player != player:
            self.advance()

        if self.viz:
            self.show()

        return copy.deepcopy(self.board)

    def advance(self):
        self.cur_player = (self.cur_player + 1) % self.num_players
        self._available_moves = None

    def available_moves(self) -> List[int]:
        if self._available_moves:
            return self._available_moves

        _, board_dead, board_live = self.board.grid[self.cur_player]
        board_special = 100 * board_dead - board_live
        avail = []
        for piece_name in self.pieces[self.cur_player]:
            for i, orient in enumerate(PIECES[piece_name].orientations):
                orient_pieces = np.flip(orient[0])
                avail_board = convolve2d(orient_pieces, board_special, mode="valid") < 0
                coords = np.nonzero(avail_board)
                avail += [
                    Move(
                        player=self.cur_player,
                        piece=piece_name,
                        orientation=i,
                        row=int(r),
                        col=int(c),
                    )
                    for (r, c) in zip(
                        coords[0].astype(np.int), coords[1].astype(np.int)
                    )
                ]
        self._available_moves = avail
        return avail

    def show(self):
        for i, player_pieces in enumerate(self.pieces):
            print(f"Player {i} pieces: {[p for p in player_pieces]}")
        self.board.show()
        plt.show()

    def to_json(self) -> bytes:
        rep = {
            "num_players": self.num_players,
            "board": self.board.to_dict(),
            "current_player": self.cur_player,
            "scores": self.scores,
            "available_moves": self.avail_moves,
        }
        return json.dumps(rep)

    @classmethod
    def from_json(cls, data: bytes):
        rep = json.loads(data)
        grid = np.array(rep["board"]["grid"])
        n, _, h, w = grid.shape
        assert n == rep["num_players"]
        blks = cls(num_players=rep["num_players"], board_h=h, board_w=w)
        blks.board.grid = grid
        return blks


class BlockusError(Exception):
    pass


def random(blokus):
    avail = blokus.available_moves()
    return avail[np.random.choice(len(avail))]


def biggest(blokus):
    avail = blokus.available_moves()
    sizes = [PIECES[move.piece].tiles for move in avail]
    biggest_avail = [move for move, size in zip(avail, sizes) if size == max(sizes)]
    return biggest_avail[np.random.choice(len(biggest_avail))]


def compete_agents(
    agents: List[Callable], n=1, verbose=True, print_moves=False, **blokus_args
):
    scores = []

    with Timer(
        granularity=Timer.MILLISECONDS,
        print_func=Timer.moving_average(lambda ms: print(f"{ms}ms"), size=10),
        disable=not verbose,
    ) as timer:
        for trial in range(n):
            blokus = Blokus(**blokus_args)
            while blokus.available_moves() != []:
                cur_player = blokus.cur_player
                move = agents[cur_player](blokus)
                if print_moves:
                    print(f"(Scores={blokus.scores})\t Placed {move}")
                blokus.place(move)
            scores.append(blokus.scores)
            timer.tick()
    return np.array(scores)


if __name__ == "__main__":
    scores = compete_agents([biggest, biggest], n=100)
    print(f"{np.mean(scores, axis=0)} +/ {np.std(scores, axis=0)}, {np.sum(scores[:, 0] > scores[:, 1]) / scores.shape[0]}")
