from deep_twist.data import utils
from shapely.geometry import Polygon


def overlap(rect1, rect2):
    poly1 = Polygon(utils.rect_to_points(rect1)) 
    print(poly1)
    poly2 = Polygon(utils.rect_to_points(rect2))
    print(poly2)
    return poly1.intersection(poly2).area / poly1.union(poly2).area


def angles_similar(angle1, angle2, thresh):
    mod_diff = (angle2 - angle1) % 360
    return mod_diff <= thresh or mod_diff >= 360 - thresh


def is_successful_grasp(rect, pos):
    for pos_rect in pos:
        is_overlapping = (overlap(rect, pos_rect) > 0.25)
        is_aligned = angles_similar(rect[2], pos_rect[2], 30)
        if is_overlapping and is_aligned:
            return True
    return False

