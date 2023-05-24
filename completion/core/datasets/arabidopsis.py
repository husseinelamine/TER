import json
import logging
import torch
import random
import multiprocessing
from functools import partial
from tqdm import tqdm
from .utils import Compose
from .dataset import Dataset
from .utils_ import get_max_file_number, rename_file, get_this_max_file_number, replace_backslash, add_ply_ext_if_needed
def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}
    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)
    return taxonomy_ids, model_ids, data

def process_sample(cfg, subset, dc, s):
    if subset == 'test' or subset == 'val':
        # Code for test or val subset
        gt_path = cfg.dataset.partial_points_path % (dc['taxonomy_id']+'/'+ s)
        n = get_this_max_file_number(gt_path)
        i = random.randint(0, n)
        obj = {
            'taxonomy_id': dc['taxonomy_id'],
            'model_id': s,
            'partial_cloud_path': rename_file(gt_path, i),
            'gtcloud_path': gt_path
        }
        return obj
    else:
        # Code for train subset
        n = get_max_file_number(cfg.dataset.partial_points_path % (dc['taxonomy_id']+'/'+ s))
        file_list = {
            'taxonomy_id': dc['taxonomy_id'],
            'model_id': s,
            'partial_cloud_path': [
                cfg.dataset.partial_points_path % (dc['taxonomy_id']+'/'+ str(i) + '.ply')
                for i in range(n)
            ],
            'gtcloud_path': cfg.dataset.partial_points_path % (dc['taxonomy_id']+'/gt.ply')
        }
        return file_list

def process_category(cfg, subset, dc):
    logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
    samples = dc[subset]
    
    # Create a partial function with fixed arguments
    process_func = partial(process_sample, cfg, subset, dc)
    
    # Use multiprocessing.Pool to parallelize the processing
    with multiprocessing.Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(process_func, samples), total=len(samples), leave=False))
    
    return results

class ADPDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.dataset.category_file_path) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.dataset.n_renderings if subset == 'train' else 1
        file_list = self._get_file_list(self.cfg, subset, n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == 'train'
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == 'train':
            return Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.dataset.n_points
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.dataset.n_points
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []
        i = 1
        dataset_length = len(self.dataset_categories)
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]
            print(f"{i} / {dataset_length}")
            i=i+1
            for s in samples:

                if subset == 'test' or subset == 'val':
                    gt_path = cfg.dataset.partial_points_path % (dc['taxonomy_id']+'/'+ s)
                    n = get_this_max_file_number(gt_path)
                    i = random.randint(0, n)
                    obj = {'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': rename_file(gt_path, i),
                    'gtcloud_path': gt_path
                    }
                    file_list.append(obj)
                else:
                    n = get_max_file_number(cfg.dataset.partial_points_path % (dc['taxonomy_id']+'/'+ s))
                    file_list.append({
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        'partial_cloud_path': [
                            cfg.dataset.partial_points_path % (dc['taxonomy_id']+'/'+ str(i) + '.ply')
                            for i in range(n)
                        ],
                        'gtcloud_path':
                            cfg.dataset.partial_points_path % (dc['taxonomy_id']+'/gt.ply'),
                    })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list
    
    def _get_file_list1(self, cfg, subset, n_renderings = 1):

        """Prepare file list for the dataset"""
        file_list = []
        for dc in tqdm(self.dataset_categories, leave=False):
            results = process_category(cfg, subset, dc)
            file_list.extend(results)
        
        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list
