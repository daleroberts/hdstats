import numpy.testing as npt
import numpy as np
import hdstats
import joblib
import pytest

import dtw as sdtw


class TestDynamicTimeWarping:
    def test_warp(self):
        x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
        y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)

        D0 = np.array(
            [
                [1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 14.0, 16.0, 16.0],
                [2.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 14.0, 16.0, 16.0],
                [2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0],
                [2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0],
                [3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 5.0],
                [6.0, 6.0, 6.0, 4.0, 4.0, 4.0, 4.0, 3.0, 5.0, 7.0],
                [7.0, 7.0, 7.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 5.0],
                [7.0, 7.0, 7.0, 5.0, 5.0, 5.0, 5.0, 6.0, 4.0, 4.0],
                [8.0, 8.0, 8.0, 5.0, 5.0, 5.0, 5.0, 6.0, 4.0, 6.0],
                [9.0, 9.0, 9.0, 7.0, 7.0, 7.0, 7.0, 8.0, 6.0, 4.0],
            ]
        )

        path0 = np.array(
            [
                [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9],
                [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9],
            ]
        )

        dist1, D1, path1 = hdstats.dtw(x, y)

        np.testing.assert_almost_equal(0.2, dist1)
        np.testing.assert_equal(D0, D1)
        np.testing.assert_equal(path0, path1)

    def test_local_warp(self):
        x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
        y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)

        D0 = np.array(
            [
                [1.0, 2.0, 3.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                [2.0, 2.0, 3.0, 5.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                [2.0, 2.0, 2.0, 3.0, 4.0, np.inf, np.inf, np.inf, np.inf, np.inf],
                [np.inf, 2.0, 2.0, 3.0, 4.0, 5.0, np.inf, np.inf, np.inf, np.inf],
                [np.inf, np.inf, 3.0, 2.0, 2.0, 2.0, 2.0, np.inf, np.inf, np.inf],
                [np.inf, np.inf, np.inf, 4.0, 4.0, 4.0, 4.0, 3.0, np.inf, np.inf],
                [np.inf, np.inf, np.inf, np.inf, 4.0, 4.0, 4.0, 4.0, 3.0, np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf, 5.0, 5.0, 6.0, 4.0, 4.0],
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 5.0, 6.0, 4.0, 6.0],
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8.0, 6.0, 4.0],
            ]
        )

        path0 = np.array(
            [
                [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9],
                [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9],
            ]
        )

        dist1, D1, path1 = hdstats.local_dtw(x, y, 2)

        np.testing.assert_equal(D0, D1)
        np.testing.assert_equal(path0, path1)

    def test_dist(self):
        x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
        y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)

        dist1 = hdstats.dtw_dist(x, y)

        np.testing.assert_almost_equal(0.2, dist1)
