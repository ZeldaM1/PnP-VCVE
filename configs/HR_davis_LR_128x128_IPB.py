exp_name = 'HR_davis_LR_128x128_IPB'
# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='IconVSR_restore_wo_refill_mv_ipb_fast_domain_dynamic_with_par',
        mid_channels=64,
        num_blocks=8,
        padding=3,
        with_cat=True,
        use_base_qp=True,
        num_experts=6,
        expert_softmax=True,
        init_weight=True,
        with_bias=True,
        with_se=True,
        with_par=True,
        one_layer=True,
        blocktype='drt',
        channel_first=True,
        sparse_val=False,
        align_key=True,
        vsr=False,
        ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
val_dataset_type = 'SRREDSMultipleGTCompressDataset'
compress_data_ratio=[0.2, 0.6, 0.85,1]
 
GT_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'base_QPs', 'QPs','partitions']),
    dict(type='PairedRandomCrop_mv', gt_patch_size=128),
    dict(type='Flip', keys=['lq', 'gt', 'mvs','partitions'], flip_ratio=0.5,direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt', 'mvs','partitions'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt', 'mvs','partitions'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'mvs','slices','base_QPs','QPs','partitions']),
    dict(type='Collect', keys=['lq', 'gt','mvs','slices','base_QPs','QPs','partitions'], meta_keys=['lq_path', 'gt_path'])
]

HR_pipeline = [
    dict(type='GenerateSegmentIndices_Mix_Compress', interval_list=[1]),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList_Mix_Compress_ipb',
        io_backend='disk',
        key='lq',
        channel_order='rgb',
        random_compress=True,
        load_mv=True,
        load_qp_slice=True,
        load_base_qp=True,
        load_partition=True,
        drconv=True,
        replace_qp_withIPB=True,
        # qp_slice_file='/data/zenghuimin/reds/train/HR/mv_REDS_train_HR_v2.json',
        qp_slice_file='dataset/REDS_train_HR/REDS_train_HR.json',
        data_ratio=compress_data_ratio),
]
HR_pipeline.extend(GT_pipeline)
LR_pipeline = [
    dict(type='GenerateSegmentIndices_Mix_Compress', interval_list=[1]),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList_Mix_Compress_ipb',
        io_backend='disk',
        key='lq',
        channel_order='rgb',
        random_compress=True,
        load_mv=True,
        load_qp_slice=True,
        load_base_qp=True,
        load_partition=True,
        drconv=True,
        replace_qp_withIPB=True,
        # qp_slice_file='/data/zenghuimin/reds/train/LR/mv_REDS_train_QP_slice_v3.json',
        qp_slice_file='dataset/REDS_train_LR/REDS_train_LR.json',
        data_ratio=compress_data_ratio),
]
LR_pipeline.extend(GT_pipeline)
davis_pipeline = [
    dict(type='GenerateSegmentIndices_Mix_Compress', filename_tmpl='{:05d}.png', interval_list=[1]),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList_Mix_Compress_ipb',
        io_backend='disk',
        key='lq',
        channel_order='rgb',
        random_compress=True,
        load_mv=True,
        load_qp_slice=True,
        load_base_qp=True,
        load_partition=True,
        drconv=True,
        replace_qp_withIPB=True,
        # qp_slice_file='/data/zenghuimin/code/davis_all/train_2017_QP_slice_all.json',
        qp_slice_file='dataset/davis_all/train_2017_QP_slice_all.json',
        data_ratio=compress_data_ratio),
]
davis_pipeline.extend(GT_pipeline)


test_pipeline = [
    dict(type='GenerateSegmentIndices_LR', interval_list=[1]),
    dict(
        type='LoadImageFromFileList_ipb',
        io_backend='disk',
        key='lq',
        channel_order='rgb',
        random_compress=False,
        load_mv=True,
        load_qp_slice=True,
        load_base_qp=True,
        load_partition=True,
        drconv=True,
        replace_qp_withIPB=True,
        qp_slice_file='dataset/REDS_test_HR/multi_cprs_REDS_test_HR.json'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'base_QPs', 'QPs','partitions']),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'mvs','slices','base_QPs','QPs','partitions']),
    dict(type='Collect', keys=['lq', 'gt', 'mvs','slices','base_QPs','QPs','partitions'],meta_keys=['lq_path', 'gt_path', 'key'])
]


data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=5, drop_last=True, mix_data=True,weights=[1,2,1],replacement=False),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=[
        dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type='SRREDSMultipleGTMixCompressDataset',
            cprs15_folder='dataset/REDS_train_HR/crf15/png',
            cprs25_folder='dataset/REDS_train_HR/crf25/png',
            cprs35_folder='dataset/REDS_train_HR/crf35/png',
            lq_folder='dataset/REDS_train_HR/crf15/png',
            gt_folder='dataset/REDS_train_HR/sharp/png',
            num_input_frames=15,
            pipeline=HR_pipeline,
            scale=1,
            val_partition='REDS4',
            test_mode=False)),
            dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type='SRREDSMultipleGTMixCompressDataset',
            cprs15_folder='dataset/REDS_train_LR/crf15/png',
            cprs25_folder='dataset/REDS_train_LR/crf25/png',
            cprs35_folder= 'dataset/REDS_train_LR/crf35/png',
            lq_folder='dataset/REDS_train_LR/crf15/png',
            gt_folder='dataset/REDS_train_LR/X4/png',
            num_input_frames=15,
            pipeline=LR_pipeline,
            scale=1,
            val_partition='REDS4',
            test_mode=False)),
            dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type='DAVISMultipleGTMixCompressDataset',
            cprs15_folder='dataset/davis_all/crf15/png',
            cprs25_folder='dataset/davis_all/crf25/png',
            cprs35_folder='dataset/davis_all/crf35/png',
            lq_folder='dataset/davis_all/crf15/png',
            gt_folder='dataset/davis_all/sharp/png',
            num_input_frames=15,
            pipeline=davis_pipeline,
            scale=1,
            test_mode=False))

    ],
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='dataset/REDS_test_HR/crf35/png',
        gt_folder='dataset/REDS_test_HR/X4/png',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=1,
        val_partition='REDS4',
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder='dataset/REDS_test_HR/crf35/png',
        gt_folder='dataset/REDS_test_HR/X4/png',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=1,
        val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=2e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

# learning policy
total_iters = 300000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=2000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
