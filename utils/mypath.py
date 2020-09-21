"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200', 'drew-fish', 'custom'}
        assert(database in db_names)

        if database == 'drew-fish':
            return '/home/ubuntu/Home/projects/unsupervised/Unsupervised-Classificatio-minen/datasets/drew-fish'

        if database == 'custom':
            return '/home/ubuntu/Home/projects/unsupervised/Unsupervised-Classificatio-minen/datasets/custom'

        if database == 'cifar-10':
            return '/home/ubuntu/Home/projects/unsupervised/Unsupervised-Classification/datasets/'

        elif database == 'cifar-20':
            return '/path/to/cifar-20/'

        elif database == 'stl-10':
            return '/path/to/stl-10/'

        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return '/path/to/imagenet/'

        else:
            raise NotImplementedError
