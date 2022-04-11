# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import transforms as T


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ],
    'DEEPFASHION2':[
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 
        18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 
        32, 31, 34, 33, 36, 35, 38, 37, 40, 39, 42, 41, 44, 43, 
        46, 45, 48, 47, 50, 49, 52, 51, 54, 53, 56, 55, 58, 57, 
        60, 59, 62, 61, 64, 63, 66, 65, 68, 67, 70, 69, 72, 71, 
        74, 73, 76, 75, 78, 77, 80, 79, 82, 81, 84, 83, 86, 85, 
        88, 87, 90, 89, 92, 91, 94, 93, 96, 95, 98, 97, 100, 99, 
        102, 101, 104, 103, 106, 105, 108, 107, 110, 109, 112, 
        111, 114, 113, 116, 115, 118, 117, 120, 119, 122, 121, 
        124, 123, 126, 125, 128, 127, 130, 129, 132, 131, 134, 
        133, 136, 135, 138, 137, 140, 139, 142, 141, 144, 143, 
        146, 145, 148, 147, 150, 149, 152, 151, 154, 153, 156, 
        155, 158, 157, 160, 159, 162, 161, 164, 163, 166, 165, 
        168, 167, 170, 169, 172, 171, 174, 173, 176, 175, 178, 
        177, 180, 179, 182, 181, 184, 183, 186, 185, 188, 187, 
        190, 189, 192, 191, 194, 193, 196, 195, 198, 197, 200, 
        199, 202, 201, 204, 203, 206, 205, 208, 207, 210, 209, 
        212, 211, 214, 213, 216, 215, 218, 217, 220, 219, 222, 
        221, 224, 223, 226, 225, 228, 227, 230, 229, 232, 231, 
        234, 233, 236, 235, 238, 237, 240, 239, 242, 241, 244, 
        243, 246, 245, 248, 247, 250, 249, 252, 251, 254, 253, 
        256, 255, 258, 257, 260, 259, 262, 261, 264, 263, 266, 
        265, 268, 267, 270, 269, 272, 271, 274, 273, 276, 275, 
        278, 277, 280, 279, 282, 281, 284, 283, 286, 285, 288, 
        287, 290, 289, 292, 291, 294
    ],
    'DEEPFASHION2_WITH_CENTER':[
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
        18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29,
        32, 31, 34, 33, 36, 35, 38, 37, 40, 39, 42, 41, 44, 43,
        46, 45, 48, 47, 50, 49, 52, 51, 54, 53, 56, 55, 58, 57,
        60, 59, 62, 61, 64, 63, 66, 65, 68, 67, 70, 69, 72, 71,
        74, 73, 76, 75, 78, 77, 80, 79, 82, 81, 84, 83, 86, 85,
        88, 87, 90, 89, 92, 91, 94, 93, 96, 95, 98, 97, 100, 99,
        102, 101, 104, 103, 106, 105, 108, 107, 110, 109, 112,
        111, 114, 113, 116, 115, 118, 117, 120, 119, 122, 121,
        124, 123, 126, 125, 128, 127, 130, 129, 132, 131, 134,
        133, 136, 135, 138, 137, 140, 139, 142, 141, 144, 143,
        146, 145, 148, 147, 150, 149, 152, 151, 154, 153, 156,
        155, 158, 157, 160, 159, 162, 161, 164, 163, 166, 165,
        168, 167, 170, 169, 172, 171, 174, 173, 176, 175, 178,
        177, 180, 179, 182, 181, 184, 183, 186, 185, 188, 187,
        190, 189, 192, 191, 194, 193, 196, 195, 198, 197, 200,
        199, 202, 201, 204, 203, 206, 205, 208, 207, 210, 209,
        212, 211, 214, 213, 216, 215, 218, 217, 220, 219, 222,
        221, 224, 223, 226, 225, 228, 227, 230, 229, 232, 231,
        234, 233, 236, 235, 238, 237, 240, 239, 242, 241, 244,
        243, 246, 245, 248, 247, 250, 249, 252, 251, 254, 253,
        256, 255, 258, 257, 260, 259, 262, 261, 264, 263, 266,
        265, 268, 267, 270, 269, 272, 271, 274, 273, 276, 275,
        278, 277, 280, 279, 282, 281, 284, 283, 286, 285, 288,
        287, 290, 289, 292, 291, 294, 294
    ],
    'DEEPFASHION2_cat1': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
       18, 17, 20, 19, 22, 21, 24, 23
    ],
    'DEEPFASHION2_cat1_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
        18, 17, 20, 19, 22, 21, 24, 23, 25
    ]
}


def build_transforms(cfg, is_train=True):
    assert is_train is True, 'Please only use build_transforms for training.'

    if 'coco' in cfg.DATASET.DATASET:
        dataset_name = 'COCO'
    elif 'crowd_pose' in cfg.DATASET.DATASET:
        dataset_name = 'CROWDPOSE'
    else:
        raise ValueError('Please implement flip_index \
            for new dataset: %s.' % cfg.DATASET.DATASET)

    if 'deepfashion2' in cfg.DATASET.ROOT:
        dataset_name = 'DEEPFASHION2'
    if 'deepfashion2_cat1' in cfg.DATASET.ROOT:
        dataset_name = 'DEEPFASHION2_cat1'

    coco_flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER']
    print('build_transform PARAMS  cfg.DATASET.INPUT_SIZE, cfg.DATASET.OUTPUT_SIZE', cfg.DATASET.INPUT_SIZE, cfg.DATASET.OUTPUT_SIZE, dataset_name)
    print('build_transform PARAMS cfg.DATASET.MAX_ROTATION,cfg.DATASET.MIN_SCALE,cfg.DATASET.MAX_SCALE, cfg.DATASET.SCALE_TYPE, cfg.DATASET.MAX_TRANSLATE',cfg.DATASET.MAX_ROTATION,cfg.DATASET.MIN_SCALE,cfg.DATASET.MAX_SCALE, cfg.DATASET.SCALE_TYPE, cfg.DATASET.MAX_TRANSLATE)
    transforms = T.Compose(
        [
            T.RandomAffineTransform(
                cfg.DATASET.INPUT_SIZE,
                cfg.DATASET.OUTPUT_SIZE,
                cfg.DATASET.MAX_ROTATION,
                cfg.DATASET.MIN_SCALE,
                cfg.DATASET.MAX_SCALE,
                cfg.DATASET.SCALE_TYPE,
                cfg.DATASET.MAX_TRANSLATE
            ),
            T.RandomHorizontalFlip(
                coco_flip_index, cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.FLIP),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    return transforms
