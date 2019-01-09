from unittest import TestCase

from matrixprofile.order import *
import numpy as np


class TestClass(TestCase):
    def test_linear_order_length(self):
        ord = LinearOrder(10)
        t = 0
        indices = []

        while t is not None:
            indices.append(t)
            t = ord.next()

        assert (len(indices[1:]) == 10)


    def test_linear_order_vals(self):
        ord = LinearOrder(10)
        t = 0
        indices = []

        while t is not None:
            indices.append(t)
            t = ord.next()

        unique_vals = np.unique(indices[1:])
        outcome = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert (unique_vals == outcome).all()


    def test_random_order_length(self):
        ord = RandomOrder(10)
        t = 0
        indices = []

        while t is not None:
            indices.append(t)
            t = ord.next()

        assert (len(indices[1:]) == 10)


    def test_random_order_vals(self):
        ord = RandomOrder(10)
        t = 0
        indices = []

        while t is not None:
            indices.append(t)
            t = ord.next()

        unique_vals = np.unique(indices[1:])
        outcome = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert (unique_vals == outcome).all()
