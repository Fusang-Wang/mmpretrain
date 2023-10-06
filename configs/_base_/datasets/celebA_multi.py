# dataset settings
dataset_type = 'MultiLabelDataset'
data_preprocessor = dict(
    num_classes=3,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    # generate onehot-format labels for multi-label classification.
    to_onehot=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=128),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=128),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root='/home/alpha/Desktop/mmpretrain_blackH_smile_lipsticks/train',
        ann_file='/home/alpha/Desktop/mmpretrain_blackH_smile_lipsticks/multi_train.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
# train_dataloader = dict(
#     ...
#     dataset=dict(
#         type='BaseDataset',
#         data_root='data',
#         ann_file='annotations/train.json',
#         data_prefix='train/',
#         pipeline=...,
#     )
# )
val_dataloader = dict(
    batch_size=16,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root='/home/alpha/Desktop/mmpretrain_blackH_smile_lipsticks/valid',
        ann_file='/home/alpha/Desktop/mmpretrain_blackH_smile_lipsticks/multi_valid.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

# test_dataloader = dict(
#     batch_size=16,
#     num_workers=6,
#     dataset=dict(
#         type=dataset_type,
#         data_root='/home/alpha/Desktop/mmpretrain_blackH_smile_lipsticks/test',
#         ann_file='/home/alpha/Desktop/mmpretrain_blackH_smile_lipsticks/multi_test_no0.json',
#         pipeline=train_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=True),
# )

test_dataloader = dict(
    batch_size=16,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root='/home/alpha/Desktop/style_mixing_test/img',
        ann_file='/home/alpha/Desktop/style_mixing_test/multi_test.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

# calculate precision_recall_f1 and mAP
val_evaluator = [
    dict(type='MultiLabelMetric'),
    dict(type='AveragePrecision')
]

# If you want standard test, please manually configure the test dataset
test_dataloader = test_dataloader
test_evaluator = val_evaluator
