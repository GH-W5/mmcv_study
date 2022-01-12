# -*- coding:UTF-8 -*-

# author: ErQ
# datetime:2022/1/12 11:35
# software: PyCharm

"""
文件说明：
    
"""
import numpy as np


def quantize(arr, min_val, max_val, levels, dtype=np.int64):
    """
    量化数组从(-inf, inf)到[0, levels-1]。为什么是[0, levels-1]，可以结合图像像素值取值范围[0, 255]
    :param arr:输入数组
    :param min_val:要裁剪的最小值
    :param max_val:要裁剪的最大值
    :param levels:量化等级
    :param dtype:被量化数组的数据类型
    :return:量化数组
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(f'levels 必须是正整数，但是获得{levels}')
    if min_val >= max_val:
        raise ValueError(f'min_val的值{min_val}必须比max_val的值{max_val}小')

    arr = np.clip(arr, min_val, max_val) - min_val  # [0, max_val-min_val]
    quantized_arr = np.minimum(
        np.floor(levels * arr / (max_val - min_val)).astype(dtype), levels - 1  # 向下取整，可以结合图像像素值取值范围[0, 255]
    )

    return quantized_arr


def dequantize(arr, min_val, max_val, levels, dtype=np.float64):
    """
    取消数组量化（量化恢复）
    :param arr:输入数组
    :param min_val:要裁剪的最小值
    :param max_val:要裁剪的最大值
    :param levels:量化等级
    :param dtype:被量化数组的数据类型
    :return:取消量化的数组
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(f'levels 必须是正整数，但是获取levels为{levels}')
    if min_val >= max_val:
        raise ValueError(f'min_val的值{min_val}必须比max_val的值{max_val}小')

    dequantized_arr = (arr + 0.5).astype(dtype) * (max_val - min_val) / levels + min_val    # 加0.5，数组中0代表的范围是[0,1)
                                                                                            # 准确来说，0到1之间的离散均值，用0.5
                                                                                            # 更好
    return dequantized_arr
