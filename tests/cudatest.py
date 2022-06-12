########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from multiprocessing import cpu_count
from torch import cuda
from unittest import TestCase
########################################################################################################################


class TestCudaPytorch(TestCase):

    def test_cuda_available(self):
        print('cuda.is_available():', cuda.is_available())
        self.assertTrue(True)

    def test_cpu_count(self):
        print('cpu_count():', cpu_count())
        self.assertTrue(True)

########################################################################################################################
