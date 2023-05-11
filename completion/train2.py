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

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./configs/pcn_cd1.yaml', help='Configuration File')
    args = parser.parse_args()
    return args

def train(config):

    train_dataloader = builder.make_dataloader(config, 'train')
    test_dataloader = builder.make_dataloader(config, config.test.split)

    
    model = builder.make_model(config)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()


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

    init_epoch = 1
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
        try:
            with tqdm(train_dataloader) as t:
                for _, (_, _, data) in enumerate(t):
                    print(data['partial_cloud'].shape) # add this line
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

    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    args = get_args_from_command_line()

    config = yaml_reader.read_yaml(args.config)


    set_seed(config.train.seed)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.train.gpu)
    try:
        train(config)
    except Exception as e:
        # play a sound to alert the error
        import winsound
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

        # log and raise
        logging.error(e)
        raise e
        
        
