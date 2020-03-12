import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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
    # F-piece
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

## for debugging: shows all orientations of each piece
def main():
    for piece_name in PIECES:
        piece = PIECES[piece_name]
        print(f"piece {piece_name}: {piece.tiles} tiles, {len(piece.orientations)} orientations")
        piece.show_orientations()
        plt.show()

if __name__ == "__main__":
    main()
