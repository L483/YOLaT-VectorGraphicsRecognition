# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from utils.svg_utils.split_cross import split_cross
from Datasets.bezier_parser import BezierParser
from Datasets.svg_parser import SVGParser
from Datasets.svg_parser import SVGGraphBuilderBezier2 as SVGGraphBuilderBezier
from svgpathtools import Path, Line, Arc
import math
import pickle
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../'))


def shape2Path(type_dict):
    # svg to Bézier curve
    parser = BezierParser()
    paths = Path()

    # line
    for start_end in type_dict['line']['start_end']:
        x0, y0, x1, y1 = start_end
        path = parser.line2BezierPath({'x1': x0, 'y1': y0, 'x2': x1, 'y2': y1})
        paths += path

    # arc
    for start_end, param in zip(type_dict['arc']['start_end'], type_dict['arc']['param']):
        start = complex(start_end[0], start_end[1])
        radius = complex(param[0], param[1])
        rotation = param[2]
        large_arc = param[3]
        sweep = param[4]
        end = complex(start_end[2], start_end[3])
        # path = Path(Arc(start, radius, rotation, large_arc, sweep, end))
        path = parser._a2c(Arc(start, radius, rotation, large_arc, sweep, end))
        paths += path

    # circle
    for param in type_dict['circle']['param']:
        cx, cy, r = param
        path = parser.circle2BezierPath({'cx': cx, 'cy': cy, 'r': r})
        paths += path

    return paths


def getConnnectedComponent(node_dict):
    edges = node_dict['edge']['shape']
    pos = node_dict['pos']['spatial']
    is_control = node_dict['attr']['is_control']
    # print(edges)
    adj = np.eye(pos.shape[0], pos.shape[0]).astype(np.bool)
    for e in edges:
        adj[e[0], e[1]] = True
        adj[e[1], e[0]] = True

    n_node = pos.shape[0]
    visited = [False if not is_control[i] else True for i in range(n_node)]
    clusters = []

    for start_node in range(0, n_node):
        if visited[start_node]:
            continue

        cluster = [start_node]
        visited[start_node] = True
        queue = [start_node]

        while len(queue) != 0:
            node_idx = queue.pop(0)
            neighbors = adj[node_idx]
            for i in range(0, n_node):
                if neighbors[i] and not visited[i]:
                    cluster.append(i)
                    visited[i] = True
                    queue.append(i)

        clusters.append(cluster)

    return clusters


def mergeCC(node_dict):
    edges = node_dict['edge']['shape']
    pos = node_dict['pos']['spatial']
    color = node_dict['attr']['color']
    is_control = node_dict['attr']['is_control']

    cc = getConnnectedComponent(node_dict)

    paths = []
    bboxs = []

    shape_shape_edges = []
    super_shape_edges = []
    offset = pos.shape[0]
    for i, cluster in enumerate(cc):
        pos_cluster = pos[cluster]
        max_x = pos_cluster[:, 0].max(0)
        min_x = pos_cluster[:, 0].min(0)
        max_y = pos_cluster[:, 1].max(0)
        min_y = pos_cluster[:, 1].min(0)
        bboxs.append((min_x, min_y, max_x, max_y))

        for ii, idx in enumerate(cluster):
            for idx_j in cluster[ii + 1:]:
                if idx == idx_j:
                    continue
                shape_shape_edges.append(tuple(sorted([idx, idx_j])))

        if True:
            real_max_x = pos_cluster[:, 0].max(0) * width
            real_min_x = pos_cluster[:, 0].min(0) * width
            real_max_y = pos_cluster[:, 1].max(0) * height
            real_min_y = pos_cluster[:, 1].min(0) * height

            p0 = complex(real_min_x, real_min_y)
            p1 = complex(real_max_x, real_min_y)
            p2 = complex(real_max_x, real_max_y)
            p3 = complex(real_min_x, real_max_y)
            paths.append(Path(Line(p0, p1),
                              Line(p1, p2),
                              Line(p2, p3),
                              Line(p3, p0)
                              ))

    cross_shape_edges = []
    same_cc = np.zeros((len(bboxs), len(bboxs))).astype(np.bool)
    for i, parent_bb in enumerate(bboxs):
        for j, child_bb in enumerate(bboxs):
            if i == j:
                continue
            inter_rect_x1 = max(parent_bb[0], child_bb[0])
            inter_rect_y1 = max(parent_bb[1], child_bb[1])
            inter_rect_x2 = min(parent_bb[2], child_bb[2])
            inter_rect_y2 = min(parent_bb[3], child_bb[3])

            child_area = (child_bb[2] - child_bb[0]) * \
                (child_bb[3] - child_bb[1])
            is_parent_child = False

            if child_area > 0:
                inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * \
                    max(inter_rect_y2 - inter_rect_y1, 0)
                if inter_area * 1.0 / child_area > 0.9:
                    is_parent_child = True
            else:
                if child_bb[2] - child_bb[0] == 0:
                    if inter_rect_x2 - inter_rect_x1 == 0 and max(inter_rect_y2 - inter_rect_y1, 0) > 0.9 * (child_bb[3] - child_bb[1]):
                        is_parent_child = True
                if child_bb[3] - child_bb[1] == 0:
                    if max(inter_rect_x2 - inter_rect_x1, 0) > 0.9 * (child_bb[2] - child_bb[0]) and inter_rect_y2 - inter_rect_y1 == 0:
                        is_parent_child = True

            if is_parent_child:  # same cc
                for parent_idx in cc[i]:
                    for child_idx in cc[j]:
                        cross_shape_edges.append(
                            tuple(sorted([parent_idx, child_idx])))
                        same_cc[i, j] = True
                        same_cc[j, i] = True

    def get_all_neighboors(root, ret):
        all_neighbors = same_cc[root]
        for i, is_neighbor in enumerate(all_neighbors):
            if i == root:
                continue
            if not is_neighbor:
                continue
            if visited[i]:
                continue
            ret.append(i)
            visited[i] = True
            get_all_neighboors(i, ret)

    visited = np.zeros(same_cc.shape[0]).astype(np.bool)
    merged_cc = []
    for i, all_neighbors in enumerate(same_cc):
        if visited[i]:
            continue
        cluster = [i]
        get_all_neighboors(i, cluster)
        merged_cc.append(cluster)
        visited[i] = True

    new_cc = []
    for cluster in merged_cc:
        t = []
        for idx in cluster:
            t += cc[idx]
        new_cc.append(t)

    def get_attr(edges):
        ret = []
        for e in edges:
            pos_start = pos[e[0]]
            pos_end = pos[e[1]]

            euc_d2 = (pos_start[0] - pos_end[0]) * (pos_start[0] - pos_end[0]) + \
                (pos_start[1] - pos_end[1]) * (pos_start[1] - pos_end[1])
            angle = (pos_start[0] - pos_end[0]) / (np.sqrt(euc_d2) + 1e-7)
            w = 1 / np.exp(euc_d2)
            # w = (w - 0.8) / 0.2
            # print(angle, w)
            if math.isnan(angle):
                print(angle, pos_start, pos_end, dot_prod, cos_theta)
                raise SystemExit

            ret.append([0, 0, 0, 0, angle, euc_d2])
        return ret

    shape_shape_edges = list(set(shape_shape_edges))
    cross_shape_edges = list(set(cross_shape_edges))
    shape_shape_edge_attr = get_attr(shape_shape_edges)
    cross_shape_edge_attr = get_attr(cross_shape_edges)

    return np.array(shape_shape_edges), np.array(cross_shape_edges), np.array(shape_shape_edge_attr), np.array(cross_shape_edge_attr), paths, new_cc


if __name__ == '__main__':
    graph_builder = SVGGraphBuilderBezier()
    input_dir = 'data/FloorPlansGraph5_iter/'
    output_dir = 'data/FloorPlansGraph5_iter/'
    dir_list = os.listdir(input_dir)

    angles = []
    distances = []
    for dir_name in dir_list:
        if not os.path.isdir(os.path.join(input_dir, dir_name)):
            continue
        svg_list = os.listdir(os.path.join(input_dir, dir_name))
        for svg_name in svg_list:
            if '.svg' not in svg_name:
                continue
            filepath = os.path.join(input_dir, dir_name, svg_name)
            print(filepath)
            p = SVGParser(filepath)
            type_dict = split_cross(p.get_all_shape())
            width, height = p.get_image_size()
            paths = shape2Path(type_dict)  # get Bezier curve
            node_dict = graph_builder.bezierPath2Graph(paths,
                                                       {'width': width,
                                                        'height': height,
                                                        'stroke': 'black',
                                                        'stroke-width': 6}
                                                       )

            # print(node_dict['edge']['shape'])
            for key in node_dict:  # pos, attr, edge, edge_attr...
                for k in node_dict[key]:
                    node_dict[key][k] = np.array(node_dict[key][k])
                    if len(node_dict[key][k].shape) == 1:
                        node_dict[key][k] = node_dict[key][k][:, None]

            # merge nodes that are close
            node_dict = graph_builder.mergeNode(node_dict)
            if True:
                e = node_dict['edge']['shape']  # edges: [[s, e]]
                for ee in e:
                    if ee[0] == ee[1]:
                        print(ee)

            # getConnnectedComponent(node_dict)
            # super_pos, super_color, shape_shape_edges, super_shape_edges, super_super_edges, bbox_paths = getSuperNode(node_dict)

            # merge connected components
            shape_shape_edges, cross_shape_edges, shape_shape_edge_attr, cross_shape_edge_attr, bbox_paths, cc = mergeCC(
                node_dict)
            bbox_paths.append(paths)

            # num of nodes
            start_end_size = node_dict['pos']['spatial'].shape[0]

            # node_dict['pos']['spatial'] = np.concatenate([node_dict['pos']['spatial'], super_pos], axis = 0)
            # node_dict['attr']['color'] = np.concatenate([node_dict['attr']['color'], super_color], axis = 0)
            # node_dict['edge']['super'] = np.concatenate([shape_shape_edges, super_shape_edges, super_super_edges], axis = 0)
            node_dict['edge']['super'] = np.concatenate(
                [shape_shape_edges, cross_shape_edges], axis=0)  # super: position-wise edge
            # node_dict['attr']['is_control'] = np.concatenate([node_dict['attr']['is_control'], np.zeros((super_pos.shape[0], 1)).astype(np.bool)], axis = 0)
            # node_dict['attr']['is_super'] = np.concatenate([np.zeros((start_end_size, 1)).astype(np.bool), np.ones((super_pos.shape[0], 1)).astype(np.bool)], axis = 0)
            node_dict['attr']['is_super'] = np.zeros(
                (start_end_size, 1)).astype(np.bool)
            node_dict['edge_attr']['super'] = np.concatenate(
                [shape_shape_edge_attr, cross_shape_edge_attr], axis=0)
            # print(node_dict['attr']['is_control'].shape, node_dict['attr']['is_super'].shape)
            node_dict['img_width'] = width
            node_dict['img_height'] = height
            node_dict['cc'] = cc

            if not os.path.isdir(os.path.join(output_dir, dir_name)):
                os.mkdir(os.path.join(output_dir, dir_name))
            # wsvg(bbox_paths, filename = os.path.join(output_dir, dir_name, svg_name))
            output_name = svg_name.replace('.svg', '.pkl')
            pickle.dump(node_dict, open(os.path.join(
                output_dir, dir_name, output_name), 'wb'))

            for a in node_dict['edge_attr']['super']:
                angles.append(a[4])
                distances.append(a[5])

    stats = {'angles': {'mean': np.mean(angles), 'std': np.std(angles)},
             'distances': {'mean': np.mean(distances), 'std': np.std(distances)}
             }
    pickle.dump(stats, open(os.path.join(output_dir, 'stats.pkl'), 'wb'))
    print(stats)

# def getSuperNode(node_dict):
#     edges = node_dict['edge']['shape']
#     pos = node_dict['pos']['spatial']
#     color = node_dict['attr']['color']
#     is_control = node_dict['attr']['is_control']

#     cc = getConnnectedComponent(node_dict)

#     paths = []
#     bboxs = []

#     shape_shape_edges = []
#     super_shape_edges = []
#     super_pos = np.zeros((len(cc), 2))
#     super_color = np.zeros((len(cc), 3))
#     offset = pos.shape[0]

#     for i, cluster in enumerate(cc):
#         pos_cluster = pos[cluster]
#         #print(pos_cluster.shape)
#         max_x = pos_cluster[:, 0].max(0)
#         min_x = pos_cluster[:, 0].min(0)
#         max_y = pos_cluster[:, 1].max(0)
#         min_y = pos_cluster[:, 1].min(0)
#         #print(min_x, min_y, max_x, max_y)
#         bboxs.append((min_x, min_y, max_x, max_y))

#         super_pos[i] = np.mean(pos_cluster, axis = 0)
#         super_color[i] = np.mean(color[cluster], axis = 0)
#         for ii, idx in enumerate(cluster):
#             super_shape_edges.append((offset + i, idx))
#             for idx_j in cluster[ii + 1, :]:
#                 if idx == idx_j: continue
#                 shape_shape_edges.append((idx, idx_j))

#         if True:
#             real_max_x = pos_cluster[:, 0].max(0) * width
#             real_min_x = pos_cluster[:, 0].min(0) * width
#             real_max_y = pos_cluster[:, 1].max(0) * height
#             real_min_y = pos_cluster[:, 1].min(0) * height

#             p0 = complex(real_min_x, real_min_y)
#             p1 = complex(real_max_x, real_min_y)
#             p2 = complex(real_max_x, real_max_y)
#             p3 = complex(real_min_x, real_max_y)
#             paths.append(Path(Line(p0, p1),
#                 Line(p1, p2),
#                 Line(p2, p3),
#                 Line(p3, p0)
#             ))

#     super_super_edges = []
#     same_cc = np.zeros((len(bboxs), len(bboxs))).astype(np.bool)
#     for i, parent_bb in enumerate(bboxs):
#         for j, child_bb in enumerate(bboxs):
#             if i == j: continue
#             inter_rect_x1 = max(parent_bb[0], child_bb[0])
#             inter_rect_y1 = max(parent_bb[1], child_bb[1])
#             inter_rect_x2 = min(parent_bb[2], child_bb[2])
#             inter_rect_y2 = min(parent_bb[3], child_bb[3])

#             child_area = (child_bb[2] - child_bb[0]) * (child_bb[3] - child_bb[1])
#             is_parent_child = False

#             if child_area > 0:
#                 inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * max(inter_rect_y2 - inter_rect_y1, 0)
#                 if inter_area * 1.0 / child_area > 0.9:
#                     is_parent_child = True
#             else:
#                 if child_bb[2] - child_bb[0] == 0:
#                     if inter_rect_x2 - inter_rect_x1 == 0 and max(inter_rect_y2 - inter_rect_y1, 0) > 0.9 * (child_bb[3] - child_bb[1]):
#                         is_parent_child = True
#                 if child_bb[3] - child_bb[1] == 0:
#                     if  max(inter_rect_x2 - inter_rect_x1, 0) > 0.9 * (child_bb[2] - child_bb[0]) and inter_rect_y2 - inter_rect_y1 == 0:
#                         is_parent_child = True

#             if is_parent_child:
#                 same_cc[i, j] = True
#                 same_cc[j, i] = True
#                 #super_super_edges.append((offset + i, offset + j))
#                 #print(parent_bb[0] * width, parent_bb[1] * height, parent_bb[2] * width, parent_bb[3] * height)
#                 #print(child_bb[0] * width, child_bb[1] * height, child_bb[2] * width, child_bb[3] * height)
#                 #print('_______________________________________')

#     return super_pos, super_color, np.array(shape_shape_edges), np.array(super_shape_edges), np.array(super_super_edges), paths
