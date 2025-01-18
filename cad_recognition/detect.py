# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from utils.det_util import non_max_suppression
import logging
from matplotlib.ticker import NullLocator
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import __init__

from config import OptInit
from utils.ckpt_util import load_pretrained_models
from torch.nn import functional as F

import os
import time
import datetime
import numpy as np
import random

from PIL import Image

import torch
from torch.utils.data import DataLoader
from train import collate

import matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":
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
    classes = [
        'armchair',
        'bed',
        'door1',
        'door2',
        'sink1',
        'sink2',
        'sink3',
        'sink4',
        'sofa1',
        'sofa2',
        'table1',
        'table2',
        'table3',
        'tub',
        'window1',
        'window2',
        'None'
    ]
    opt.in_channels = test_dataset[0].x.shape[1]

    logging.info('===> Loading the network ...')
    if opt.arch == 'centernet3cc_rpn_gp_iter2':
        from architecture3cc_rpn_gp_iter2 import SparseCADGCN, DetectionLoss

    model = SparseCADGCN(opt).to(opt.device)
    total_params = sum(p.numel() for p in model.parameters())
    print('number of params', total_params / 1000000)

    # if opt.multi_gpus:
    #     model = DataParallel(SparseDeepGCN(opt)).to(opt.device)
    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(
        model, opt.pretrained_model, opt.phase)
    logging.info(model)

    model.eval()  # Set in evaluation mode

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")

    mean_inference_time = 0
    with torch.no_grad():
        sample_metrics = []
        labels = []
        starter, ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)

        for i_batch, (data, slices) in enumerate(test_loader):
            prev_time = time.time()

            pos_slice = slices['pos']
            for key in slices:
                if 'edge' in key:
                    s = slices[key]
                    # print(key, s)
                    o = getattr(data, key)
                    for i_s in range(0, len(s) - 1):
                        start = s[i_s]
                        end = s[i_s + 1]
                        o[start:end] += pos_slice[i_s]
                    setattr(data, key, o)
                elif key == 'bbox_idx':
                    bbox_offset = slices['labels']
                    s = slices[key]
                    # print(key, s)
                    o = getattr(data, key)
                    for i_s in range(0, len(s) - 1):
                        start = s[i_s]
                        end = s[i_s + 1]
                        o[start:end] += bbox_offset[i_s]
                    setattr(data, key, o)

            if opt.arch == 'centernet':
                pred_cls, pred_coord, pred_coord_max = model.predict(data.x.cuda(),
                                                                     [data.edge.cuda().T, data.edge_control.cuda(
                                                                     ).T, data.edge_pos.cuda().T],
                                                                     [None, None, data.e_weight_pos.cuda(
                                                                     ).T]
                                                                     )
            else:
                torch.cuda.synchronize()
                st = time.time()
                out = model.predict(data, slices)
                torch.cuda.synchronize()
                et = time.time()
                mean_inference_time += et - st
                # print('overal inference', et - st)

                pred_cls = out[0]
                pred_coord = out[1]
                pred_coord_max = out[-1]

            if pred_coord_max is not None:
                pred_coord = pred_coord_max

            if opt.arch == 'centernet3cc_rpn_iter' or 'centernet3cc_rpn_gp_iter' in opt.arch:
                slices['bbox'] = out[4]
                # print(out[0].size(), data.labels.size())
                # print(data.labels.size(), data.has_obj.size())

            # print(slices)
            image_slice = slices['x']
            label_slice = slices['gt_labels']

            for i in range(0, len(image_slice) - 1):
                if 'centernet3cc_rpn' not in opt.arch:
                    start = image_slice[i]
                    end = image_slice[i + 1]
                    is_control_mask = ~data.is_control[start:end].squeeze()
                    pos_img = data.pos[start:end][is_control_mask].cuda()

                    pred_coord_img = pred_coord[start:end][is_control_mask]
                    pred_cls_img = pred_cls[start:end][is_control_mask]
                else:
                    t_start = slices['bbox'][i]
                    t_end = slices['bbox'][i + 1]
                    pred_coord_img = pred_coord[t_start:t_end]
                    pred_cls_img = pred_cls[t_start:t_end]

                if opt.arch == 'centernet3cc':
                    not_super = ~data.is_super[start:end][is_control_mask].squeeze(
                    )
                    pred_cls_img = pred_cls_img[not_super]
                    pred_coord_img = pred_coord_img[not_super]
                    pos_img = pos_img[not_super]

                # print('before', pred_coord_img, data.width[i], data.height[i])
                pred_coord_img[:, 0] *= data.width[i]
                pred_coord_img[:, 2] *= data.width[i]
                pred_coord_img[:, 1] *= data.height[i]
                pred_coord_img[:, 3] *= data.height[i]
                # pos_img[:, 0] *= data.width[i]
                # pos_img[:, 1] *= data.height[i]

                # pred_coord_img[:, 0:2] = pos_img - pred_coord_img[:, 0:2]
                # pred_coord_img[:, 2:4] = pos_img + pred_coord_img[:, 2:4]
                if 'centernet3cc_rpn' not in opt.arch:
                    p0 = pred_coord_img[:, 0:2] - pred_coord_img[:, 2:4] / 2.0
                    p1 = pred_coord_img[:, 0:2] + pred_coord_img[:, 2:4] / 2.0
                    pred_coord_img[:, 0:2] = p0
                    pred_coord_img[:, 2:4] = p1

                # pred_coord_img[:, 0:2] = pos_img
                # pred_coord_img[:, 2:4] = pos_img

                pred_cls_img = F.softmax(pred_cls_img, dim=1)
                # pred_cls_img = torch.cat((torch.ones(pred_cls_img.size(0), 1).cuda(), pred_cls_img), dim = 1)

                if 'centernet3cc_rpn' not in opt.arch:
                    pred_cls_img = torch.cat(
                        (pred_cls_img.max(1, keepdim=True)[0], pred_cls_img), dim=1)
                else:
                    pred_cls_img = torch.cat(
                        (1 - pred_cls_img[:, -1][:, None], pred_cls_img[:, 0:-1]), dim=1)
                pred = torch.cat(
                    (pred_coord_img, pred_cls_img), dim=1).unsqueeze(0)
                # print(pred_coord_img.size(), pred_cls_img.size(), pred.size())

                torch.cuda.synchronize()
                st = time.time()
                detections = non_max_suppression(
                    pred, conf_thres=0.75, nms_thres=0.5)
                torch.cuda.synchronize()
                et = time.time()
                mean_inference_time += et - st
                print('nms time:', et - st)

                '''    
                start = image_slice[i]
                end = image_slice[i + 1]
                gt_coord_img = data.bbox[start:end][is_control_mask]
                gt_cls_img = data.labels[start:end][is_control_mask].unsqueeze(1)
                gt_coord_img[:, 0] *= data.width[i]
                gt_coord_img[:, 2] *= data.width[i]
                gt_coord_img[:, 1] *= data.height[i]
                gt_coord_img[:, 3] *= data.height[i]
                detections = [torch.cat((gt_coord_img, torch.ones((gt_cls_img.size(0), 2)), gt_cls_img), 1)]
                detections = [x.cpu().numpy() for x in detections]
                '''

                # Log progress
                current_time = time.time()
                inference_time = datetime.timedelta(
                    seconds=current_time - prev_time)

                # print("\t+ Batch %d-%d, Inference Time: %s" % (i_batch, i, inference_time))

                # Save image and detections
                detections = [x.cpu().numpy() for x in detections]
                imgs.extend([data.filepath[i]])
                img_detections.extend(detections)
                prev_time = time.time()

        print('mean inference time', mean_inference_time /
              len(test_loader.dataset))
        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = Image.open(path.replace('svg', 'tiff')).convert(mode='RGB')
            plt.figure()
            fig, ax = plt.subplots(1)
            fig.set_size_inches(30, 20)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                # detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = np.array(
                    list(set(detections[:, -1])), dtype=np.int)
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print("\t+ Label: %s, Conf: %.5f" %
                          (classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(
                        np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle(
                        (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    t = plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)] + ' ' + '%.2f' % cls_conf,
                        color="white",
                        verticalalignment="bottom",
                        bbox={"color": color, "pad": 0},
                        fontsize=25,
                        weight="bold"
                    )
#                    t.set_position((x1, y1 - bbox.y1 - bbox.y0))

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())

            filename = os.path.dirname(path).split(
                '/')[-1] + '_' + os.path.basename(path).split(".")[0]

            output_path = os.path.join("output_all", f"{filename}.png")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()
