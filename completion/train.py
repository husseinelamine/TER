# -*- coding: utf-8 -*-
# @Author: XP

import os
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import helpers, average_meter, scheduler, yaml_reader, loss_util, misc
from core import builder
from test import test
from configup import update_yaml_file 
import subprocess                                       
import random
import http.client, urllib

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='/kaggle/working/TER/completion/configs/arabidopsis.yaml', help='Configuration File')
    args = parser.parse_args()
    return args

def train(config):


    train_dataloader = builder.make_dataloader(config, 'train')
    test_dataloader = builder.make_dataloader(config, config.test.split)

    
    model = builder.make_model(config)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()


    # out folders
    if not config.train.out_path:
        config.train.out_path = './exp'
    output_dir = os.path.join(config.train.out_path, '%s', datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
    config.train.path_checkpoints = output_dir % 'checkpoints'
    config.train.path_logs = output_dir % 'logs'
    if not os.path.exists(config.train.path_checkpoints):
        os.makedirs(config.train.path_checkpoints)
 
    # log writers
    train_writer = SummaryWriter(os.path.join(config.train.path_logs, 'train'))
    val_writer = SummaryWriter(os.path.join(config.train.path_logs, 'test'))

    init_epoch = config.train.init_epoch
    best_metric = float('inf')
    steps = 0

    if config.train.resume:
        if not os.path.exists(config.train.model_path):
            raise Exception('checkpoints does not exists: {}'.format(config.test.model_path))

        print('Recovering from %s ...' % (config.train.model_path), end='')
        checkpoint = torch.load(config.test.model_path)
        model.load_state_dict(checkpoint['model'])
        print('recovered!')

        init_epoch = checkpoint['epoch_index']
        best_metric = checkpoint['best_metric']
        optimizer = builder.make_optimizer(config, model, init_epoch)
    else:
        optimizer = builder.make_optimizer(config, model)
    scheduler = builder.make_schedular(config, optimizer, last_epoch=init_epoch if config.train.resume else -1)

    multiplier = 1.0
    if config.test.loss_func == 'cd_l1':
        multiplier = 1e3
    elif config.test.loss_func == 'cd_l2':
        multiplier = 1e4
    elif config.test.loss_func == 'emd':
        multiplier = 1e2

    completion_loss = loss_util.Completionloss(loss_func=config.train.loss_func)
    n_batches = len(train_dataloader)
    avg_meter_loss = average_meter.AverageMeter(['loss_partial', 'loss_pc', 'loss_p1', 'loss_p2', 'loss_p3'])
    for epoch_idx in range(init_epoch, config.train.epochs):
        avg_meter_loss.reset()
        model.train()
        #torch.cuda.empty_cache()

        try:
            with train_dataloader as t:
                for batch_idx, (_, _, data) in enumerate(t):
                    logging.info('Epoch: %d/%d | Iter: %d/%d' % (epoch_idx, config.train.epochs, batch_idx + 1, n_batches))
                    if config.dataset.name in ['PCN', 'Completion3D', 'Arabidopsis']:
                        for k, v in data.items():
                            data[k] = helpers.var_or_cuda(v)
                        partial = data['partial_cloud']
                        gt = data['gtcloud']
                    elif config.dataset.name in ['ShapeNet-34', 'ShapeNet-Unseen21']:
                        npoints = config.dataset.n_points
                        gt = data.cuda()
                        partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                                                            fixed_points=None)
                        partial = partial.cuda()
                    else:
                        raise Exception('Unknown dataset: {}'.format(config.dataset.name))

                    pcds_pred = model(partial)
                    loss_total, losses = completion_loss.get_loss(pcds_pred, partial, gt)

                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()
                    losses = [ls*multiplier for ls in losses]
                    avg_meter_loss.update(losses)
                    n_itr =  epoch_idx * n_batches + batch_idx
                    train_writer.add_scalar('Loss/Batch/partial_matching', losses[0], n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_pc', losses[1], n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_p1', losses[2], n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_p2', losses[3], n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_p3', losses[4], n_itr)

                    t.set_description(
                        '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, config.train.epochs, batch_idx + 1, n_batches))
                    t.set_postfix(
                        loss='%s' % ['%.4f' % l for l in losses])
                    # trying to free memory
                    # torch.cuda.empty_cache()
                    # each 300 batches send a notification
                    if batch_idx % 300 == 0:
                        conn = http.client.HTTPSConnection("api.pushover.net:443")
                        conn.request("POST", "/1/messages.json",
                        urllib.parse.urlencode({
                            "token": "adv35dbsktbcgkdez9mufr6im6t68d",
                            "user": "uo6ey6gx5wps3u4yzswaq7casvh7wk",
                            "message": f"Epoch: {epoch_idx} | Iter: {batch_idx} | Loss: {losses[4]}",
                        }), { "Content-type": "application/x-www-form-urlencoded" })
                        r = conn.getresponse()
                        conn.close()
        except Exception as e:
            raise e

        scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        train_writer.add_scalar('Loss/Epoch/partial_matching', avg_meter_loss.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_meter_loss.avg(1), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_meter_loss.avg(2), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_meter_loss.avg(3), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p3', avg_meter_loss.avg(4), epoch_idx)


        cd_eval = test(config, model=model, test_dataloader=test_dataloader, validation=True,
                       epoch_idx=epoch_idx, test_writer=val_writer, completion_loss=completion_loss)

                     
        configp = yaml_reader.read_yaml(config.config_path)
        # Save checkpoints
        if epoch_idx % config.train.save_freq == 0 or cd_eval < best_metric:
            file_name = 'ckpt-best.pth' if cd_eval < best_metric else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(config.train.path_checkpoints, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metric': best_metric,
                'model': model.state_dict()
            }, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metric:
                best_metric = cd_eval
        """if epoch_idx % config.train.split_freq == 0:
            cmd = f"python C:/Users/Husse/Documents/TER/TERCompletionNuages/testing/split7.py --limit-dir 4 --limit-plant 700 --seed {random.randint(30, 1000)}"
            subprocess.run(cmd, shell=True)
            config_path = config.config_path
            config_checkpoints = config.train.path_checkpoints
            config_logs = config.train.path_logs
            config = configp
            config.config_path = config_path
            config.train.path_checkpoints = config_checkpoints
            config.train.path_logs = config_logs
            train_dataloader = builder.make_dataloader(config, 'train')
            test_dataloader = builder.make_dataloader(config, config.test.split)"""
        if epoch_idx % config.train.optimizer.decay_freq == 0:
            optimizer = builder.make_optimizer(config, model, epoch_idx)
            scheduler = builder.make_schedular(config, optimizer, last_epoch=epoch_idx)

        config.train.optimizer.decay_factor = configp.train.optimizer.decay_factor
        config.train.optimizer.decay_freq = configp.train.optimizer.decay_freq
        config.train.optimizer.decay_ignore = configp.train.optimizer.decay_ignore
        config.train.optimizer.kwargs.lr = configp.train.optimizer.kwargs.lr
        config.train.save_freq = configp.train.save_freq
        config.train.split_freq = configp.train.split_freq

    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    args = get_args_from_command_line()

    config = yaml_reader.read_yaml(args.config)

    config.config_path = args.config

    set_seed(config.train.seed)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.train.gpu)
    try:
        train(config)
    except Exception as e:
        
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
        urllib.parse.urlencode({
            "token": "adv35dbsktbcgkdez9mufr6im6t68d",
            "user": "uo6ey6gx5wps3u4yzswaq7casvh7wk",
            "message": e,
        }), { "Content-type": "application/x-www-form-urlencoded" })
        r = conn.getresponse()
        print(r.status, r.reason)
        conn.close()
        # log and raise
        logging.error(e)
        raise e
