import os
import torch
import shutil
from collections import OrderedDict
import logging
import numpy as np


# ismax means max best
def load_pretrained_models(model, pretrained_model, phase, ismax=True):
    if ismax:
        best_value = -np.inf
    else:
        best_value = np.inf
    epoch = -1

    if pretrained_model:
        if os.path.isfile(pretrained_model):
            logging.info(
                "===> Loading checkpoint '{}'".format(pretrained_model))
            checkpoint = torch.load(pretrained_model)
            try:
                best_value = checkpoint['best_value']
                if best_value == -np.inf or best_value == np.inf:
                    show_best_value = False
                else:
                    show_best_value = True
            except:
                # best_value = best_value
                show_best_value = False

            model_dict = model.state_dict()
            ckpt_model_state_dict = checkpoint['state_dict']

            # rename ckpt (avoid name is not same because of multi-gpus)
            is_model_multi_gpus = True if list(
                model_dict)[0][0][0] == 'm' else False
            is_ckpt_multi_gpus = True if list(ckpt_model_state_dict)[
                0][0] == 'm' else False

            if is_model_multi_gpus != is_ckpt_multi_gpus:
                temp_dict = OrderedDict()
                for k, v in ckpt_model_state_dict.items():
                    if is_ckpt_multi_gpus:
                        name = k[7:]  # remove 'module.'
                    else:
                        name = 'module.'+k  # add 'module'
                    temp_dict[name] = v
                # load params
                ckpt_model_state_dict = temp_dict

            model_dict.update(ckpt_model_state_dict)
            model.load_state_dict(ckpt_model_state_dict)

            if show_best_value:
                logging.info("The pretrained_model is at checkpoint {}. \t "
                             "Best value: {}".format(checkpoint['epoch'], best_value))
            else:
                logging.info("The pretrained_model is at checkpoint {}.".format(
                    checkpoint['epoch']))

            if phase == 'train':
                epoch = checkpoint['epoch']
            else:
                epoch = -1
        else:
            raise ImportError(
                "===> No checkpoint found at '{}'".format(pretrained_model))
    else:
        logging.info('===> No pre-trained model')
    return model, best_value, epoch


def load_pretrained_optimizer(pretrained_model, optimizer, scheduler, lr, use_ckpt_lr=True):
    if pretrained_model and os.path.isfile(pretrained_model):
        checkpoint = torch.load(pretrained_model)
        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        if 'scheduler_state_dict' in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if use_ckpt_lr:
                try:
                    lr = scheduler.get_lr()[0]
                except:
                    # lr = lr
                    pass

    return optimizer, scheduler, lr


def save_checkpoint(state, is_best, save_path, postname):
    filename = '{}/{}_{}.pth'.format(save_path, postname, int(state['epoch']))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_best.pth'.format(save_path, postname))
