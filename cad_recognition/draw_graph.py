# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
import __init__

from config import OptInit
from Datasets.svg import SESYDFloorPlan


import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset


import logging

if __name__ == "__main__":
    opt = OptInit().get_args()
    logging.info('===> Creating dataloader ...')

    test_dataset = SESYDFloorPlan(
        opt.data_dir, pre_transform=T.NormalizeScale(), partition='train')
    test_loader = DataLoader(test_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=8,
                             collate_fn=InMemoryDataset.collate)

    with torch.no_grad():
        for i_batch, (data, slices) in enumerate(test_loader):

            print(slices)
            image_slice = slices['x']
            label_slice = slices['gt_labels']
            edge_slice = slices['edges']
            raise SystemExit
