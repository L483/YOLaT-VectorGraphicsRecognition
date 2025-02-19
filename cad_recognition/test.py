# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import __init__
from torch.utils.data import DataLoader
import logging

from config import OptInit
from utils.ckpt_util import load_pretrained_models
from utils.metrics import AverageMeter
from train import collate
# from  Datasets.svg import SESYDFloorPlan as CADDataset

from train import test


def main():
    opt = OptInit().get_args()
    logging.info('===> Creating dataloader ...')

    if opt.graph == 'bezier':
        from Datasets.svg import SESYDFloorPlan as CADDataset
    elif opt.graph == 'shape':
        from Datasets.svg2 import SESYDFloorPlan as CADDataset
    elif opt.graph == 'bezier_edge_attr':
        from Datasets.svg3 import SESYDFloorPlan as CADDataset
    elif opt.graph == 'bezier_cc_bb_iter':
        from Datasets.graph_dict3 import SESYDFloorPlan as CADDataset

    test_dataset = CADDataset(
        opt.data_dir, opt, partition=opt.phase, data_aug=False, do_mixup=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=8,
                             collate_fn=collate)

#    if opt.multi_gpus:
#        train_loader = DataListLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
#    else:
#        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    opt.n_classes = len(list(test_dataset.class_dict.keys()))
    # classes = list(range(0, opt.n_classes))
    opt.in_channels = test_dataset[0].x.shape[1]

    logging.info('===> Loading the network ...')

    if opt.arch == 'centernet3cc_rpn_gp_iter2':
        from architecture3cc_rpn_gp_iter2 import SparseCADGCN, DetectionLoss

    model = SparseCADGCN(opt).to(opt.device)
    criterion = DetectionLoss(opt)
    # if opt.multi_gpus:
    #     model = DataParallel(SparseDeepGCN(opt)).to(opt.device)
    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(
        model, opt.pretrained_model, opt.phase)
    logging.info(model)

    # imgs = []  # Stores image paths
    # img_detections = []  # Stores detections for each image index

    opt.test_values = AverageMeter()
    opt.test_value = 0.
    test(model, test_loader, criterion, opt)


if __name__ == '__main__':
    main()
