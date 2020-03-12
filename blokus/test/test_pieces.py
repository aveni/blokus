import numpy as np
import unittest
from ..pieces import PIECES

class TestPieces(unittest.TestCase):
    def test_pieces_count(self):
        # 1 of size 1
        # 1 of size 2
        # 2 of size 3
        # 5 of size 4
        # 12 of size 5
        self.assertEqual(len(PIECES), 21)

    def test_num_orientations(self):
        self.assertEqual(len(PIECES["F"].orientations), 8)
        self.assertEqual(len(PIECES["I1"].orientations), 1)
        self.assertEqual(len(PIECES["I2"].orientations), 2)
        self.assertEqual(len(PIECES["I3"].orientations), 2)
        self.assertEqual(len(PIECES["I4"].orientations), 2)
        self.assertEqual(len(PIECES["I5"].orientations), 2)
        self.assertEqual(len(PIECES["L4"].orientations), 8)
        self.assertEqual(len(PIECES["L5"].orientations), 8)
        self.assertEqual(len(PIECES["N"].orientations), 8)
        self.assertEqual(len(PIECES["T4"].orientations), 4)
        self.assertEqual(len(PIECES["T5"].orientations), 4)
        self.assertEqual(len(PIECES["O"].orientations), 1)
        self.assertEqual(len(PIECES["P"].orientations), 8)
        self.assertEqual(len(PIECES["U"].orientations), 4)
        self.assertEqual(len(PIECES["V3"].orientations), 4)
        self.assertEqual(len(PIECES["V5"].orientations), 4)
        self.assertEqual(len(PIECES["W"].orientations), 4)
        self.assertEqual(len(PIECES["X"].orientations), 1)
        self.assertEqual(len(PIECES["Y"].orientations), 8)
        self.assertEqual(len(PIECES["Z4"].orientations), 4)
        self.assertEqual(len(PIECES["Z5"].orientations), 4)

    def test_tiles(self):
        self.assertEqual(PIECES["F"].tiles, 5)
        self.assertEqual(PIECES["I1"].tiles, 1)
        self.assertEqual(PIECES["I2"].tiles, 2)
        self.assertEqual(PIECES["I3"].tiles, 3)
        self.assertEqual(PIECES["I4"].tiles, 4)
        self.assertEqual(PIECES["I5"].tiles, 5)
        self.assertEqual(PIECES["L4"].tiles, 4)
        self.assertEqual(PIECES["L5"].tiles, 5)
        self.assertEqual(PIECES["N"].tiles, 5)
        self.assertEqual(PIECES["T4"].tiles, 4)
        self.assertEqual(PIECES["T5"].tiles, 5)
        self.assertEqual(PIECES["O"].tiles, 4)
        self.assertEqual(PIECES["P"].tiles, 5)
        self.assertEqual(PIECES["U"].tiles, 5)
        self.assertEqual(PIECES["V3"].tiles, 3)
        self.assertEqual(PIECES["V5"].tiles, 5)
        self.assertEqual(PIECES["W"].tiles, 5)
        self.assertEqual(PIECES["X"].tiles, 5)
        self.assertEqual(PIECES["Y"].tiles, 5)
        self.assertEqual(PIECES["Z4"].tiles, 4)
        self.assertEqual(PIECES["Z5"].tiles, 5)

    def test_check_all_orientations(self):
        # each "orientation" is a triple of three bit-grids:
        # first is the piece itself (pieces from any player cannot overlap these squares)
        # second is the piece + squares that share an edge with the piece (further pieces from this player cannot overlap these dead squares)
        # third is the "live" squares that touch the piece only at its corners (further pieces from this player must overlap one of these squares)
        for orientation in PIECES["F"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 14)
            self.assertEqual(np.sum(orientation[2]), 7)
        for orientation in PIECES["I1"].orientations:
            self.assertEqual(np.sum(orientation[0]), 1)
            self.assertEqual(np.sum(orientation[1]), 5)
            self.assertEqual(np.sum(orientation[2]), 4)
        for orientation in PIECES["I2"].orientations:
            self.assertEqual(np.sum(orientation[0]), 2)
            self.assertEqual(np.sum(orientation[1]), 8)
            self.assertEqual(np.sum(orientation[2]), 4)
        for orientation in PIECES["I3"].orientations:
            self.assertEqual(np.sum(orientation[0]), 3)
            self.assertEqual(np.sum(orientation[1]), 11)
            self.assertEqual(np.sum(orientation[2]), 4)
        for orientation in PIECES["I4"].orientations:
            self.assertEqual(np.sum(orientation[0]), 4)
            self.assertEqual(np.sum(orientation[1]), 14)
            self.assertEqual(np.sum(orientation[2]), 4)
        for orientation in PIECES["I5"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 17)
            self.assertEqual(np.sum(orientation[2]), 4)
        for orientation in PIECES["L4"].orientations:
            self.assertEqual(np.sum(orientation[0]), 4)
            self.assertEqual(np.sum(orientation[1]), 13)
            self.assertEqual(np.sum(orientation[2]), 5)
        for orientation in PIECES["L5"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 16)
            self.assertEqual(np.sum(orientation[2]), 5)
        for orientation in PIECES["N"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 15)
            self.assertEqual(np.sum(orientation[2]), 6)
        for orientation in PIECES["T4"].orientations:
            self.assertEqual(np.sum(orientation[0]), 4)
            self.assertEqual(np.sum(orientation[1]), 12)
            self.assertEqual(np.sum(orientation[2]), 6)
        for orientation in PIECES["T5"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 15)
            self.assertEqual(np.sum(orientation[2]), 6)
        for orientation in PIECES["O"].orientations:
            self.assertEqual(np.sum(orientation[0]), 4)
            self.assertEqual(np.sum(orientation[1]), 12)
            self.assertEqual(np.sum(orientation[2]), 4)
        for orientation in PIECES["P"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 14)
            self.assertEqual(np.sum(orientation[2]), 5)
        for orientation in PIECES["U"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 15)
            self.assertEqual(np.sum(orientation[2]), 5)
        for orientation in PIECES["V3"].orientations:
            self.assertEqual(np.sum(orientation[0]), 3)
            self.assertEqual(np.sum(orientation[1]), 10)
            self.assertEqual(np.sum(orientation[2]), 5)
        for orientation in PIECES["V5"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 16)
            self.assertEqual(np.sum(orientation[2]), 5)
        for orientation in PIECES["W"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 14)
            self.assertEqual(np.sum(orientation[2]), 7)
        for orientation in PIECES["X"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 13)
            self.assertEqual(np.sum(orientation[2]), 8)
        for orientation in PIECES["Y"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 15)
            self.assertEqual(np.sum(orientation[2]), 6)
        for orientation in PIECES["Z4"].orientations:
            self.assertEqual(np.sum(orientation[0]), 4)
            self.assertEqual(np.sum(orientation[1]), 12)
            self.assertEqual(np.sum(orientation[2]), 6)
        for orientation in PIECES["Z5"].orientations:
            self.assertEqual(np.sum(orientation[0]), 5)
            self.assertEqual(np.sum(orientation[1]), 15)
            self.assertEqual(np.sum(orientation[2]), 6)


if __name__ == '__main__':
    unittest.main()
