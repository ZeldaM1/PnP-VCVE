# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
from mmedit.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--testdir_lr',default=None, type=str, help='user-specified dir')
    parser.add_argument('--testdir_gt',default=None, type=str, help='user-specified dir')
    parser.add_argument('--cascade',action='store_true',help='whether to preprocess cascaded tasks e.g., + VSR/enhance/')
    parser.add_argument('--cascade_ckpt', help='checkpoint file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    
    if args.testdir_lr is not None:
        cfg.data.test["lq_folder"]=args.testdir_lr
        print('-------------------- test LR dir :', args.testdir_lr)
    if args.testdir_gt is not None:
        cfg.data.test["gt_folder"]=args.testdir_gt
        print('-------------------- test GT dir :', args.testdir_gt)
    dataset = build_dataset(cfg.data.test)
 

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.cascade:
        assert args.cascade_ckpt is not None
        VSRmodel = build_model(cfg.VSRmodel, train_cfg=None, test_cfg=cfg.test_cfg)


    args.save_image = args.save_path is not None
    empty_cache = cfg.get('empty_cache', False)
    if not distributed:
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        if args.cascade:
            _ = load_checkpoint(VSRmodel, args.cascade_ckpt, map_location='cpu')
            VSRmodel = MMDataParallel(VSRmodel, device_ids=[0])
            model = [model,VSRmodel]

        outputs = single_gpu_test(
            model,
            data_loader,
            save_path=args.save_path,
            save_image=args.save_image)
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if args.cascade:
            VSRmodel = DistributedDataParallelWrapper(
            VSRmodel,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

        device_id = torch.cuda.current_device()
        _ = load_checkpoint(
            model,
            args.checkpoint,
            map_location=lambda storage, loc: storage.cuda(device_id))
        if args.cascade:
            _ = load_checkpoint(
            VSRmodel,
            args.cascade_ckpt,
            map_location=lambda storage, loc: storage.cuda(device_id))
            model=[model,VSRmodel]
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            save_path=args.save_path,
            save_image=args.save_image,
            empty_cache=empty_cache)

    if rank == 0 and 'eval_result' in outputs[0]:
        print('')
        # print metrics
        stats = dataset.evaluate(outputs)
        for stat in stats:
            print('Eval-{}: {}'.format(stat, stats[stat]))
        
        print('{:.4f}/{:.4f}'.format(float(stats['PSNR']), float(stats['SSIM'])))

        # save result pickle
        if args.out:
            print('writing results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)


if __name__ == '__main__':
    main()
