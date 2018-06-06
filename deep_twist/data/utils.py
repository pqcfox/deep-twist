import numpy as np
import torch
from skimage import draw


def point_mid(a, b):
    return ((a[0] + b[0])/2, (a[1] + b[1])/2) 


def point_dist(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return np.sqrt(dx**2 + dy**2)


def point_angle(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return np.degrees(np.arctan2(dy, dx)) + 90


def points_to_rect(points):
    x, y = point_mid(points[0], points[2])
    theta = point_angle(points[1], points[2])
    w = point_dist(points[0], points[1])
    h = point_dist(points[1], points[2])
    return x, y, theta, w, h


def rect_to_points(rect):
    x, y, theta, w, h = rect.numpy()
    theta_rad = np.radians(theta)
    points = []
    base_points = [(-w, h), (w, h), (w, -h), (-w, -h)]
    for base_point in base_points:
        point = (np.cos(theta_rad) * base_point[0] -
                 np.sin(theta_rad) * base_point[1] + x,
                 np.sin(theta_rad) * base_point[0] + 
                 np.cos(theta_rad) * base_point[1] + y)
        points.append(point)
    return points


def draw_rectangle(rgb, rect):
    points = rect_to_points(rect)
    for i in range(len(points)):
        point_from = points[i]
        point_to = points[(i + 1) % 4]
        rr, cc = draw.line(int(point_from[1]), int(point_from[0]), 
                                int(point_to[1]), int(point_to[0]))
        pairs = [(r, c) for r, c in zip(rr, cc) if r < rgb.shape[0] and c < rgb.shape[1]]
        rr, cc = [list(l) for l in zip(*pairs)]
        rgb[rr, cc, 0] = 0 if i % 2 == 0 else 255
        rgb[rr, cc, 1:] = 255 if i % 2 == 0 else 0
    return rgb
        

def parse_rects(rect_path, id):
    rects = [] 
    with open(rect_path) as f:
        lines = f.read().splitlines()
        for i in range(0, len(lines), 4):
            rect_lines = lines[i:(i + 4)]
            points = []
            valid = True
            for rect_line in rect_lines:
                point_values = rect_line.strip().split(' ')
                point = tuple(float(value) for value in point_values)
                if np.any(np.isnan(point)):
                    valid = False
                points.append(point)
            if not valid:
                continue

            rect = points_to_rect(points)
            rects.append(rect)
    return rects


def parse_depth(depth_path, depth_shape):
    depth = np.zeros(depth_shape)
    width = depth.shape[1]
    with open(depth_path) as f:
        for line in f.read().splitlines():
            if line[0].isdigit():
                values = line.split(' ')
                z, depth_idx = float(values[2]), int(values[4])
                row, col = depth_idx // width, depth_idx % width
                depth[row, col] = z
    return depth


def discretize_theta(theta, ticks=19):
    return (np.rint((theta % 360) * ticks / 360) % ticks).long()


def one_hot_to_rects(theta, rect_coords):
    rects = []
    for i in range(theta.size(0)):
        theta_val = torch.argmax(theta[i, :])
        rect = torch.FloatTensor((rect_coords[i, 0],
                                  rect_coords[i, 1],
                                  theta_val,
                                  rect_coords[i, 2],
                                  rect_coords[i, 3]))
        rects.append(rect)
    return rects
