# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import os
import numpy as np
from xml.dom.minidom import parse, Node

from torch_geometric.data import Data
from Datasets.svg_parser import SVGGraphBuilderShape as SVGGraphBuilder
from Datasets.svg_parser import SVGParser
from sklearn.metrics.pairwise import euclidean_distances

# from a2c import a2c


class SESYDFloorPlan(torch.utils.data.Dataset):
    def __init__(self, root, opt, partition='train', data_aug=False):
        super(SESYDFloorPlan, self).__init__()

        svg_list = open(os.path.join(
            root, partition + '_list.txt')).readlines()
        svg_list = [os.path.join(root, line.strip()) for line in svg_list]
        self.graph_builder = SVGGraphBuilder()
        # print(svg_list)

        self.pos_edge_th = opt.pos_edge_th
        self.data_aug = data_aug

        self.svg_list = svg_list

        self.class_dict = {
            'armchair': 0,
            'bed': 1,
            'door1': 2,
            'door2': 3,
            'sink1': 4,
            'sink2': 5,
            'sink3': 6,
            'sink4': 7,
            'sofa1': 8,
            'sofa2': 9,
            'table1': 10,
            'table2': 11,
            'table3': 12,
            'tub': 13,
            'window1': 14,
            'window2': 15
        }

        # self.anchors = self.get_anchor()
        '''
        self.n_objects = 0
        for idx in range(len(self.svg_list)):
            filepath = self.svg_list[idx]
            print(filepath)
            p = SVGParser(filepath)
            width, height = p.get_image_size()
            #graph_dict = self.graph_builder.build_graph(p.get_all_shape())

            gt_bbox, gt_labels = self._get_bbox(filepath, width, height)
            self.n_objects += gt_bbox.shape[0]
        print(self.n_objects)
        '''
        self.n_objects = 13238

    def __len__(self):
        return len(self.svg_list)

    def _get_bbox(self, path, width, height):
        dom = parse(path.replace('.svg', '.xml'))
        root = dom.documentElement

        nodes = []
        for tagname in ['a', 'o']:
            nodes += root.getElementsByTagName(tagname)

        bbox = []
        labels = []
        for node in nodes:
            for n in node.childNodes:
                if n.nodeType != Node.ELEMENT_NODE:
                    continue
                x0 = float(n.getAttribute('x0')) / width
                y0 = float(n.getAttribute('y0')) / height
                x1 = float(n.getAttribute('x1')) / width
                y1 = float(n.getAttribute('y1')) / height
                label = n.getAttribute('label')
                bbox.append((x0, y0, x1, y1))
                labels.append(self.class_dict[label])

        return np.array(bbox), np.array(labels)

    def gen_y(self, graph_dict, bbox, labels, width, height):
        pos = graph_dict['pos']

        th = 1e-3
        gt_bb = []
        gt_cls = []
        gt_object = []
        for node_idx, p in enumerate(pos):

            diff_0 = p[None, :] - bbox[:, 0:2]
            diff_1 = p[None, :] - bbox[:, 2:]
            in_object = (diff_0[:, 0] >= -th) & (diff_0[:, 1]
                                                 >= -th) & (diff_1[:, 0] <= th) & (diff_1[:, 1] <= th)

            object_index = np.nonzero(in_object)[0]
            if len(object_index) > 1:
                # print(object_index)
                # print('node', p[0] * width, p[1] * height, 'is inside more than one object')
                candidates = bbox[object_index]
                s = euclidean_distances(p[None, :], candidates[:, 0:2])[0]
                # print(np.argsort(s))
                object_index = object_index[np.argsort(s)]
                # print(candidates, s, object_index)
            elif len(object_index) == 0:
                # print(diff_0 * [width, height], diff_1* [width, height])
                # print(object_index)
                print('node', p[0] * width, p[1] *
                      height, 'outside all object')
                # for i, line in enumerate(bbox[:, 0:2] * [width, height]):
                #    print(i, line)
                raise SystemExit
            cls = labels[object_index[0]]
            bb = bbox[object_index[0]]
            '''
            h = bb[3] - bb[1]
            w = bb[2] - bb[0]
            offset_x = bb[0] - p[0]
            offset_y = bb[1] - p[1]
            gt_bb.append((offset_x, offset_y, w, h))
            '''
            gt_bb.append(bb)
            gt_cls.append(cls)
            gt_object.append(object_index[0])

        return np.array(gt_bb), np.array(gt_cls), np.array(gt_object)

    def __transform__(self, pos, scale, angle, translate):
        scale_m = np.eye(2)
        scale_m[0, 0] = scale
        scale_m[1, 1] = scale

        rot_m = np.eye(2)
        rot_m[0, 0:2] = [np.cos(angle), np.sin(angle)]
        rot_m[1, 0:2] = [-np.sin(angle), np.cos(angle)]

        # print(pos.shape, scale_m[0:2].shape)
        # pos = np.matmul(pos, scale_m[0:2])
        # print(pos.shape)
        center = np.array((0.5, 0.5))[None, :]
        pos -= center
        pos = np.matmul(pos, rot_m[0:2])
        pos += center
        # pos += np.array(translate)[None, :]
        return pos

    def __transform_bbox__(self, bbox, scale, angle, translate):
        p0 = bbox[:, 0:2]
        p2 = bbox[:, 2:]
        p1 = np.concatenate([p2[:, 0][:, None], p0[:, 1][:, None]], axis=1)
        p3 = np.concatenate([p0[:, 0][:, None], p2[:, 1][:, None]], axis=1)

        p0 = self.__transform__(p0, scale, angle, translate)
        p1 = self.__transform__(p1, scale, angle, translate)
        p2 = self.__transform__(p2, scale, angle, translate)
        p3 = self.__transform__(p3, scale, angle, translate)

        def bound_rect(p0, p1, p2, p3):
            x = np.concatenate(
                (p0[:, 0][:, None], p1[:, 0][:, None], p2[:, 0][:, None], p3[:, 0][:, None]), axis=1)
            y = np.concatenate(
                (p0[:, 1][:, None], p1[:, 1][:, None], p2[:, 1][:, None], p3[:, 1][:, None]), axis=1)
            x_min = x.min(1, keepdims=True)
            x_max = x.max(1, keepdims=True)
            y_min = y.min(1, keepdims=True)
            y_max = y.max(1, keepdims=True)

            return np.concatenate([x_min, y_min, x_max, y_max], axis=1)
        return bound_rect(p0, p1, p2, p3)

    def random_transfer(self, pos, bbox, gt_bbox):
        rng = np.random.default_rng()
        scale = rng.random() * 0.1 + 0.9
        angle = rng.random() * np.pi * 2
        translate = [0, 0]
        translate[0] = rng.random() * 0.2 - 0.1
        translate[1] = rng.random() * 0.2 - 0.1

        pos = self.__transform__(pos, scale, angle, translate)
        bbox = self.__transform_bbox__(bbox, scale, angle, translate)
        gt_bbox = self.__transform_bbox__(gt_bbox, scale, angle, translate)

        return pos, bbox, gt_bbox

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # for idx in range(len(self.svg_list)):
        filepath = self.svg_list[idx]
        p = SVGParser(filepath)
        width, height = p.get_image_size()
        graph_dict = self.graph_builder.build_graph(p.get_all_shape())

        gt_bbox, gt_labels = self._get_bbox(filepath, width, height)
        bbox, labels, gt_object = self.gen_y(
            graph_dict, gt_bbox, gt_labels, width, height)

        feats = graph_dict['f']
        pos = graph_dict['pos']
        is_control = np.zeros((pos.shape[0], 1))

        edge = graph_dict['edge']

        feats = torch.tensor(feats, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        edge = torch.tensor(edge, dtype=torch.long)
        is_control = torch.tensor(is_control, dtype=torch.bool)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        gt_object = torch.tensor(gt_object, dtype=torch.long)

        e_weight = torch.tensor(graph_dict['edge_weight'], dtype=torch.float32)

        # print('bbox', bbox.size())
        # print('labels', labels.size())
        # raise SystemExit

        data = Data(x=feats, pos=pos)
        data.edge = edge
        # data.edge_control = None
        # data.edge_pos = None
        data.is_control = is_control
        data.bbox = bbox
        data.labels = labels
        data.gt_bbox = gt_bbox
        data.gt_labels = gt_labels
        data.gt_object = gt_object
        data.filepath = filepath
        data.width = width
        data.height = height
        data.e_weight = e_weight

        return data


if __name__ == '__main__':
    svg_list = open(
        '/home/xinyangjiang/Datasets/SESYD/FloorPlans/train_list.txt').readlines()
    svg_list = ['/home/xinyangjiang/Datasets/SESYD/FloorPlans/' + line.strip()
                for line in svg_list]
    builder = SVGGraphBuilder()
    for line in svg_list:
        print(line)
        # line = '/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-01/file_56.svg'
        p = SVGParser(line)
        builder.build_graph(p.get_all_shape())

    # train_dataset = SESYDFloorPlan(opt.data_dir, pre_transform=T.NormalizeScale())
    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    # for batch in train_loader:
    #    pass

    # paths, attributes, svg_attributes = svg2paths2('/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-05/file_47.svg')
    # print(paths, attributes, svg_attributes)
