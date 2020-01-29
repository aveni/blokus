import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d
from typing import List, Tuple, Callable
import json
from typing import NamedTuple
import copy
from blokus.profiler import Timer


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


class Piece:
    def __init__(self, name, arr, rsymmetric, fsymmetric):
        self.name = name
        self.arr = arr
        self.rsymmetric = rsymmetric
        self.fsymmetric = fsymmetric
        self.tiles = np.sum(self.arr == 1)
        self.set_orientations()

    def set_orientations(self):
        info = np.array([self.arr == 1, self.arr > 0, self.arr < 0], dtype="float32")

        self.orientations = []
        bases = [info] if self.fsymmetric else [info, np.flip(info, axis=1)]
        for b in bases:
            for k in range(self.rsymmetric):
                self.orientations.append(np.rot90(b, k=k, axes=(1, 2)))

    def show(self, axis=None):
        if axis == None:
            fig, axis = plt.subplots(1, 1, figsize=(2, 2))
        axis.imshow(self.arr, cmap=piece_cm)
        axis.set_title(self.name)

    def show_orientations(self):
        num_o = len(self.orientations)
        fig, axes = plt.subplots(1, num_o, figsize=(2 * num_o, 2))
        if num_o == 1:
            axes = [axes]
        for info, axis in zip(self.orientations, axes):
            img = 2 * info[1] - info[0] - info[2]
            axis.imshow(img, cmap=piece_cm)
            axis.set_title(self.name)


PIECES = {
    # N-piece
    "F": Piece(
        name="F",
        arr=np.array(
            [
                [-1, 2, -1, 0, 0],
                [2, 1, 2, 2, -1],
                [2, 1, 1, 1, 2],
                [-1, 2, 1, 2, -1],
                [0, -1, 2, -1, 0],
            ]
        ),
        rsymmetric=4,
        fsymmetric=False,
    ),
    # I-pieces
    "I1": Piece(
        name="I1",
        arr=np.array([[-1, 2, -1], [2, 1, 2], [-1, 2, -1]]),
        rsymmetric=1,
        fsymmetric=True,
    ),
    "I2": Piece(
        name="I2",
        arr=np.array([[-1, 2, -1], [2, 1, 2], [2, 1, 2], [-1, 2, -1]]),
        rsymmetric=2,
        fsymmetric=True,
    ),
    "I3": Piece(
        name="I3",
        arr=np.array([[-1, 2, -1], [2, 1, 2], [2, 1, 2], [2, 1, 2], [-1, 2, -1]]),
        rsymmetric=2,
        fsymmetric=True,
    ),
    "I4": Piece(
        name="I4",
        arr=np.array(
            [[-1, 2, -1], [2, 1, 2], [2, 1, 2], [2, 1, 2], [2, 1, 2], [-1, 2, -1]]
        ),
        rsymmetric=2,
        fsymmetric=True,
    ),
    "I5": Piece(
        name="I5",
        arr=np.array(
            [
                [-1, 2, -1],
                [2, 1, 2],
                [2, 1, 2],
                [2, 1, 2],
                [2, 1, 2],
                [2, 1, 2],
                [-1, 2, -1],
            ]
        ),
        rsymmetric=2,
        fsymmetric=True,
    ),
    # L-pieces
    "L4": Piece(
        name="L4",
        arr=np.array(
            [[-1, 2, -1, 0], [2, 1, 2, 0], [2, 1, 2, -1], [2, 1, 1, 2], [-1, 2, 2, -1]]
        ),
        rsymmetric=4,
        fsymmetric=False,
    ),
    "L5": Piece(
        name="L5",
        arr=np.array(
            [
                [-1, 2, -1, 0],
                [2, 1, 2, 0],
                [2, 1, 2, 0],
                [2, 1, 2, -1],
                [2, 1, 1, 2],
                [-1, 2, 2, -1],
            ]
        ),
        rsymmetric=4,
        fsymmetric=False,
    ),
    # N-piece
    "N": Piece(
        name="N",
        arr=np.array(
            [
                [-1, 2, -1, 0],
                [2, 1, 2, -1],
                [2, 1, 1, 2],
                [-1, 2, 1, 2],
                [0, 2, 1, 2],
                [0, -1, 2, -1],
            ]
        ),
        rsymmetric=4,
        fsymmetric=False,
    ),
    # T-pieces
    "T4": Piece(
        name="T4",
        arr=np.array(
            [[-1, 2, 2, 2, -1], [2, 1, 1, 1, 2], [-1, 2, 1, 2, -1], [0, -1, 2, -1, 0]]
        ),
        rsymmetric=4,
        fsymmetric=True,
    ),
    "T5": Piece(
        name="T5",
        arr=np.array(
            [
                [-1, 2, 2, 2, -1],
                [2, 1, 1, 1, 2],
                [-1, 2, 1, 2, -1],
                [0, 2, 1, 2, 0],
                [0, -1, 2, -1, 0],
            ]
        ),
        rsymmetric=4,
        fsymmetric=True,
    ),
    # O-piece
    "O": Piece(
        name="O",
        arr=np.array([[-1, 2, 2, -1], [2, 1, 1, 2], [2, 1, 1, 2], [-1, 2, 2, -1]]),
        rsymmetric=1,
        fsymmetric=True,
    ),
    # P-piece
    "P": Piece(
        name="P",
        arr=np.array(
            [[-1, 2, 2, -1], [2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 2, -1], [-1, 2, -1, 0]]
        ),
        rsymmetric=4,
        fsymmetric=False,
    ),
    # U-piece
    "U": Piece(
        name="U",
        arr=np.array(
            [[-1, 2, -1, 2, -1], [2, 1, 2, 1, 2], [2, 1, 1, 1, 2], [-1, 2, 2, 2, -1]]
        ),
        rsymmetric=4,
        fsymmetric=True,
    ),
    # V-pieces
    "V3": Piece(
        name="V3",
        arr=np.array([[-1, 2, -1, 0], [2, 1, 2, -1], [2, 1, 1, 2], [-1, 2, 2, -1]]),
        rsymmetric=4,
        fsymmetric=True,
    ),
    "V5": Piece(
        name="V5",
        arr=np.array(
            [
                [-1, 2, -1, 0, 0],
                [2, 1, 2, 0, 0],
                [2, 1, 2, 2, -1],
                [2, 1, 1, 1, 2],
                [-1, 2, 2, 2, -1],
            ]
        ),
        rsymmetric=4,
        fsymmetric=True,
    ),
    # W-piece
    "W": Piece(
        name="W",
        arr=np.array(
            [
                [-1, 2, -1, 0, 0],
                [2, 1, 2, -1, 0],
                [2, 1, 1, 2, -1],
                [-1, 2, 1, 1, 2],
                [0, -1, 2, 2, -1],
            ]
        ),
        rsymmetric=4,
        fsymmetric=True,
    ),
    # X-piece
    "X": Piece(
        name="X",
        arr=np.array(
            [
                [0, -1, 2, -1, 0],
                [-1, 2, 1, 2, -1],
                [2, 1, 1, 1, 2],
                [-1, 2, 1, 2, -1],
                [0, -1, 2, -1, 0],
            ]
        ),
        rsymmetric=1,
        fsymmetric=True,
    ),
    # Y-piece
    "Y": Piece(
        name="Y",
        arr=np.array(
            [
                [-1, 2, 2, 2, 2, -1],
                [2, 1, 1, 1, 1, 2],
                [-1, 2, 1, 2, 2, -1],
                [0, -1, 2, -1, 0, 0],
            ]
        ),
        rsymmetric=4,
        fsymmetric=False,
    ),
    # Z-pieces
    "Z4": Piece(
        name="Z4",
        arr=np.array(
            [[-1, 2, 2, -1, 0], [2, 1, 1, 2, -1], [-1, 2, 1, 1, 2], [0, -1, 2, 2, -1]]
        ),
        rsymmetric=2,
        fsymmetric=False,
    ),
    # Z-pieces
    "Z5": Piece(
        name="Z5",
        arr=np.array(
            [
                [-1, 2, 2, -1, 0],
                [2, 1, 1, 2, 0],
                [-1, 2, 1, 2, -1],
                [0, 2, 1, 1, 2],
                [0, -1, 2, 2, -1],
            ]
        ),
        rsymmetric=2,
        fsymmetric=False,
    ),
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

        for i in range(self.num_players):
            player_image = self.grid[i, 2] * -1
            player_image += self.grid[i, 1] * 2
            player_image -= self.grid[i, 0]
            axes[i + 1].imshow(player_image, cmap=piece_cm)

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
    compete_agents([biggest, biggest], n=1000)
