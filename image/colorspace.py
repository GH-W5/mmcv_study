# -*- coding:UTF-8 -*-

# author: ErQ
# datetime:2022/1/12 15:05
# software: PyCharm

"""
文件说明：
    
"""
import cv2
import numpy as np


def imconvert(img, src, dst):
    """
    将图像从src颜色空间转换到dst颜色空间
    :param img:输入图像
    :param src:原始颜色空间，例如：rgb, hsv
    :param dst:目的颜色空间，例如：rgb, hsv
    :return:转换后的图像
    """
    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')
    out_img = cv2.cvtColor(img, code)
    return out_img


def bgr2gray(img, keepdim=False):
    """
    BGR图像转换为灰度图像
    :param img:输入图像
    :param keepdim:默认为false，返回二维灰度图像，其他，返回三维灰度图像
    :return:返回灰度图像
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


def rgb2gray(img, keepdim=False):
    """
    RGB图像转换为灰度图像
    :param img: 输入图像
    :param keepdim: 默认为false，返回二维灰度图像，其他，返回三维灰度图像
    :return: 返回灰度图像
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


def gray2bgr(img):
    """
    灰度图像转BGR图像
    :param img: 输入图像
    :return: 返回BGR图像
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return out_img


def gray2rgb(img):
    """
    灰度图像转RGB图像
    :param img: 输入图像
    :return: 返回RGB图像
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return out_img


def _convert_input_type_range(img):
    """
    改变输入图像的数据类型和范围。
    它是转换输入图数据类型np.float32,范围[0, 1]的图像。
    它主要用于rgb2ycbcr和ycbrc2rgb等颜色空间转换函数中的输入图像的预处理。
    :param img: 输入图像，接受：1.数值类型np.uint8，范围[0,255];2.数值类型np.float32，范围[0, 1]
    :return: 返回数值类型为np.float32,范围[0, 1]的转换后的图像。
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255
    else:
        raise ValueError(f'img的类型应该为np.float32或者np.uint8,但是获得的类型为{img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """
    根据dst_type转换图像的数值类型和范围
    :param img: 要转换的图像是np.float32,范围[0, 255]
    :param dst_type: (np.uint8 | np.float32): 如果是np.uint8，转换后图像类型np.uint8,范围[2, 255];如果是np.float32,转换后图像类型np.float32,范围[0,1]
    :return: 返回具有所需类型和范围的图像
    """
    if dst_type not in [np.uint8, np.float32]:
        raise ValueError(f'dst_type应该是np.uint8或者np.float32,但是获得是{dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255
    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
    """
    RGB图像转ycbcr图像
    这个函数生产的结果与MATLAB的rgb2ycbcr函数的结果相同。
    它是依据ITU-R BT.601标准定义进行的转换。更多细节参考：https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    它不同于cv2.cvtColor: `RGB <-> YCrCb`中的类似函数。
    在OpenCV中，它实现一个JPEG转换，更多细节：https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    :param img: 输入数据，它接受：1.np.uint8类型，范围[0, 255];2.np.float32类型，范围[0, 1]
    :param y_only: 是否仅返回Y通道，默认False
    :return: 输入转换为YCbCr的图像，输出图像和输入图像数据类型和范围相同。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 122.0], [128.553, -74.203, -93.786],
                  [24.966, 112.0, -18.214]]
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """
    BGR图像转ycbcr图像
    这个函数生产的结果与MATLAB的bgr2ycbcr函数的结果相同。
    它是依据ITU-R BT.601标准定义进行的转换。更多细节参考：https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    它不同于cv2.cvtColor: `BGR <-> YCrCb`中的类似函数。
    在OpenCV中，它实现一个JPEG转换，更多细节：https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    :param img: 输入数据，它接受：1.np.uint8类型，范围[0, 255];2.np.float32类型，范围[0, 1]
    :param y_only: 是否仅返回Y通道，默认False
    :return: 输入转换为YCbCr的图像，输出图像和输入图像数据类型和范围相同。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    """
    ycbcr图像转RGB图像
    这个函数生产的结果与MATLAB的ycbcr2rgb函数的结果相同。
    它是依据ITU-R BT.601标准定义进行的转换。更多细节参考：https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    它不同于cv2.cvtColor: `YCrCb <-> RGB`中的类似函数。
    在OpenCV中，它实现一个JPEG转换，更多细节：https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    :param img: 输入数据，它接受：1.np.uint8类型，范围[0, 255];2.np.float32类型，范围[0, 1]
    :param y_only: 是否仅返回Y通道，默认False
    :return: 输入转换为YCbCr的图像，输出图像和输入图像数据类型和范围相同。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    out_img = np.matmul(
        img, [[0.00456621, 0.00456621, 0.00456621],
              [0, -0.00153632, 0.00791071],
              [0.00625893, -0.00318811, 0]]
    ) * 255 + [-222.921, 135.576, -276.836]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):
    """
    ycbcr图像转BGR图像
    这个函数生产的结果与MATLAB的ycbcr2bgr函数的结果相同。
    它是依据ITU-R BT.601标准定义进行的转换。更多细节参考：https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    它不同于cv2.cvtColor: `YCrCb <-> BGR`中的类似函数。
    在OpenCV中，它实现一个JPEG转换，更多细节：https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    :param img: 输入数据，它接受：1.np.uint8类型，范围[0, 255];2.np.float32类型，范围[0, 1]
    :param y_only: 是否仅返回Y通道，默认False
    :return: 输入转换为YCbCr的图像，输出图像和输入图像数据类型和范围相同。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    out_img = np.matmul(
        img, [[0.00456621, 0.00456621, 0.00456621],
              [0.00791071, -0.00153632, 0],
              [0, -0.00318811, 0.00625893]]
    ) * 255.0 + [-276.836, 135.576, -222.921]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def convert_color_factory(src, dst):
    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""
    将一张{src.upper()}图像转换为{dst.upper()}图像
    
    Args:
        img: 输入图像
    Returns: 转换为{dst.upper()}的图像
    
    """
    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

bgr2hls = convert_color_factory('bgr', 'hls')

hls2bgr = convert_color_factory('hls', 'bgr')
