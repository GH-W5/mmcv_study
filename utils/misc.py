# -*- coding:UTF-8 -*-

# author: ErQ
# datetime:2022/1/12 17:40
# software: PyCharm

"""
文件说明：
    
"""
import collections.abc
import functools
import itertools
import subprocess
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec
from itertools import repeat


# 来自PyTorch内部组件
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def is_str(x):
    """
    输入x是否为str类型
    :param x:
    :return:
    """
    return isinstance(x, str)


def inport_modules_from_strings(imports, allow_failed_imports=False):
    """

    :param imports:
    :param allow_failed_imports:
    :return:
    """