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
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/Datasets/Geometry_Descriptor/fm/celeba_raw/test/train',
        ann_file='/mnt/Datasets/Geometry_Descriptor/fm/celeba_raw/test/train.json',
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
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/Datasets/Geometry_Descriptor/fm/celeba_raw/test/valid',
        ann_file='/mnt/Datasets/Geometry_Descriptor/fm/celeba_raw/test/valid.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/Datasets/Geometry_Descriptor/fm/celeba_raw/test/test',
        ann_file='/mnt/Datasets/Geometry_Descriptor/fm/celeba_raw/test/test.json',
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
