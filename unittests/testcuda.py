########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from multiprocessing import cpu_count
from torch import cuda, randn, device
from unittest import TestCase
########################################################################################################################


class TestCudaPytorch(TestCase):

    def test_cuda_available(self):
        if not cuda.is_available():
            print('No cuda on this machine')
            return
        cuda0 = 'cuda:0'
        x = randn(3, 4, 5, device=cuda0)
        self.assertEqual(0, x.get_device())
        cuda0 = device('cuda:0')
        x = randn(3, 4, 5, device=cuda0)
        self.assertEqual(0, x.get_device())
        x = randn(3, 4, 5)
        x = x.to(cuda0)
        self.assertEqual(0, x.get_device())

    def test_cpu_count(self):
        print('cpu_count():', cpu_count())
        self.assertTrue(True)

########################################################################################################################
