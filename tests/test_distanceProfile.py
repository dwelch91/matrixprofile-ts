from unittest import TestCase

from matrixprofile.distance_profile import *
import numpy as np


class TestClass(TestCase):
    def test_naive_distance_profile_self(self):
        outcome = (np.array([0.0, 2.828, np.inf, np.inf, np.inf, np.inf, np.inf, 2.828, 0.0]), np.array([4., 4., 4., 4., 4., 4., 4., 4., 4.]))
        b = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        assert (np.round(naive_distance_profile(b, 4, 4), 3) == outcome).all()

        # Need to confirm that we're not updating the original variable via shared memory
        assert (b == np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])).all()


    def test_naive_distance_profile_tsa_tsb(self):
        outcome = (np.array([0.0, 2.828, 4.0, 2.828, 0.0, 2.828, 4.0, 2.828, 0.0]), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        a = np.array([0.0, 1.0, 1.0, 0.0])
        b = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])

        assert (np.round(naive_distance_profile(a, 0, 4, b), 3) == outcome).all()
        assert (a == np.array([0.0, 1.0, 1.0, 0.0])).all()


    def test_mass_distance_profile_self(self):
        outcome = (np.array([0.0, 2.828, np.inf, np.inf, np.inf, np.inf, np.inf, 2.828, 0.0]), np.array([4., 4., 4., 4., 4., 4., 4., 4., 4.]))
        b = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        assert (np.round(mass_distance_profile(b, 4, 4), 3) == outcome).all()

        # Need to confirm that we're not updating the original variable via shared memory
        assert (b == np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])).all()


    def test_mass_distance_profile_tsa_tsb(self):
        outcome = (np.array([0.0, 2.828, 4.0, 2.828, 0.0, 2.828, 4.0, 2.828, 0.0]), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        a = np.array([0.0, 1.0, 1.0, 0.0])
        b = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])

        assert (np.round(mass_distance_profile(a, 0, 4, b), 3) == outcome).all()
        assert (a == np.array([0.0, 1.0, 1.0, 0.0])).all()


    def test_stomp_distance_profile_self(self):
        outcome = (np.array([0.0, 2.828, np.inf, np.inf, np.inf, np.inf, np.inf, 2.828, 0.0]), np.array([4., 4., 4., 4., 4., 4., 4., 4., 4.]))
        ts_a = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        idx = 4
        m = 4
        ts_b = None
        dot_first = np.array([2., 1., 0., 1., 2., 1., 0., 1., 2.])
        dp = np.array([1., 0., 1., 2., 1., 0., 1., 2., 1.])
        mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result, out = stomp_distance_profile(ts_a, idx, m, ts_b, dot_first, dp, mean, std)
        assert (np.round(result, 3) == outcome).all()

        # Need to confirm that we're not updating the original variable via shared memory
        assert (ts_a == np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])).all()
