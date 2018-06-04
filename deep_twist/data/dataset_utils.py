import numpy as np
import torch

def point_mid(a, b):
    return ((a[0] + b[0])/2, (a[1] + b[1])/2)


def point_dist(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return np.sqrt(dx**2 + dy**2)


def point_angle(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return np.degrees(np.arctan2(dy, dx)) + 90


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

            x, y = point_mid(points[0], points[2])
            theta = point_angle(points[1], points[2])
            w = point_dist(points[0], points[1])
            h = point_dist(points[1], points[2])
            rects.append(torch.FloatTensor([x, y, theta, w, h]))
    return rects
