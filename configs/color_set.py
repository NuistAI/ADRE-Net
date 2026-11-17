
HCMS_COLOR_SET = {
    0: [  0,   0,   0],
    1: [173, 216, 230],
    2: [  0, 255, 255],
    3: [  0, 128,   0],
    4: [255, 255,   0],
    5: [255, 165,   0],
    6: [255,   0,   0],
    7: [139,   0,   0],
    8: [  0, 255,   0],
    9: [  0,   0,   0]
}
DUKE_COLOR_SET = {
    0: [  0,   0,   0],
    1: [173, 216, 230],
    2: [153, 210, 207],
    3: [ 73, 145, 110],
    4: [237, 203, 105],
    5: [226, 153, 92],
    6: [190, 82, 91],
    7: [112, 82, 126],
    8: [  0,   0,   0],
    9: [214, 39, 119],
}


AROI_COLOR_SET = {
    0: [  0,   0,   0],
    1: [173, 216, 230],
    2: [153, 210, 207],
    3:  [255,255,255],
    4: [0, 0, 0],
    5: [ 190, 82, 91],
    6: [237, 203, 105],
    7: [214, 39, 119],
}


def get_color_set(dataset_name: str):
    """
    Return the color map dictionary for a dataset shorthand.
    :param dataset_name: 'HCMS' or 'AROI' (case-insensitive)
    :return: Dict[int, List[int]]
    """
    color_map = {
        'hcms': HCMS_COLOR_SET,
        'aroi': AROI_COLOR_SET,
        'duke': DUKE_COLOR_SET,
    }
    return color_map[dataset_name.lower()]