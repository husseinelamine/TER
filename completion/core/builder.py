import sys
import torch
from torch.optim.lr_scheduler import StepLR

sys.path.append('../..')
from utils.misc import build_lambda_sche, build_lambda_bnsche
from models.model_completion import SnowflakeNet


def make_dataloader(config, split):
    dataset_name = config.dataset.name

    if split == 'train':
        batch_size = config.train.batch_size
        num_workers = config.train.num_workers
    else:
        batch_size = config.test.batch_size
        num_workers = config.test.num_workers

    if dataset_name == 'PCN':
        from .datasets.pcn import PCNDataLoader, collate_fn
        dataset = PCNDataLoader(config).get_dataset(split)
    elif dataset_name == 'Completion3D':
        from .datasets.c3d import C3DDataLoader, collate_fn
        dataset = C3DDataLoader(config).get_dataset(split)
    elif dataset_name == 'ShapeNet-34' or dataset_name == 'ShapeNet-Unseen21':
        from .datasets.shapenet55 import ShapeNet
        dataset = ShapeNet(config, split)
        collate_fn = None
    elif dataset_name == 'Arabidopsis':
        from .datasets.arabidopsis import ADPDataLoader, collate_fn
        dataset = ADPDataLoader(config).get_dataset(split)
    else:
        raise Exception('dataset {} not supported yet!'.format(dataset_name))
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=True,
                                            shuffle=split=='train',
                                            drop_last=False)

    return data_loader


def make_model(config):
    model = SnowflakeNet(
        dim_feat=config.model.dim_feat,
        num_pc=config.model.num_pc,
        num_p0=config.model.num_p0,
        radius=config.model.radius,
        bounding=config.model.bounding,
        up_factors=config.model.up_factors,
    )

    # if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model).cuda()

    return model


def make_optimizer(config, model, current_epoch=0):
    opti_config = config.train.optimizer
    if opti_config.type == 'Adam':
        if not current_epoch == 0:
            initial_weight_decay = opti_config.kwargs.weight_decay  # Initial weight decay value
            weight_decay_increase_factor = opti_config.decay_factor  # Factor by which weight decay increases
            weight_decay_increase_epochs = opti_config.decay_freq  # Number of epochs after which weight decay increases
            start_weight_decay_increase_epoch = opti_config.decay_ignore  # Epoch to start increasing weight decay
            epoch = current_epoch
            def get_current_weight_decay():
                if epoch >= start_weight_decay_increase_epoch:
                    # Calculate the weight decay for epochs after start_weight_decay_increase_epoch
                    decay_factor = ((epoch - start_weight_decay_increase_epoch) // weight_decay_increase_epochs)
                    #decay_factor = weight_decay_increase_factor ** ((epoch - start_weight_decay_increase_epoch) // weight_decay_increase_epochs)
                    return initial_weight_decay if decay_factor == 0 else initial_weight_decay * decay_factor
                    # Return the initial weight decay value for epochs before start_weight_decay_increase_epoch
                return initial_weight_decay

            opti_config.kwargs.weight_decay = round(get_current_weight_decay(), 4)
        print('optimizer weight decay: ', "{:.4f}".format(opti_config.kwargs.weight_decay))
        optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': opti_config.kwargs.lr}],
            **opti_config.kwargs
        )
    elif opti_config.type == 'AdamW':
        optimizer = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': opti_config.kwargs.lr}],
                                      **opti_config.kwargs)

    else:
        raise Exception('optimizer {} not supported yet!'.format(opti_config.type))


    return optimizer


def make_schedular(config, optimizer, last_epoch=-1):
    sche_config = config.train.scheduler
    if sche_config.type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=sche_config.kwargs.decay_step, gamma=sche_config.kwargs.gamma, last_epoch=last_epoch)
    elif sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    else:
        raise Exception('scheduler {} not supported yet!'.format(sche_config.type))

    return scheduler
