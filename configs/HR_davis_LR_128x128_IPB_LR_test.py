_base_ = [
    'HR_davis_LR_128x128_IPB.py'
]

# dataset settings
val_dataset_type = 'SRREDSMultipleGTCompressDataset'
 
 

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
        qp_slice_file='dataset/REDS_test_LR/REDS_test_LR.json',
        ),
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

    test=dict(
        type=val_dataset_type,
        lq_folder='dataset/REDS_test_LR/crf15/png',
        gt_folder='dataset/REDS_test_LR/X4/png',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=1,
        val_partition='REDS4',
        test_mode=True),
)
