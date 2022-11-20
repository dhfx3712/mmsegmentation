数据集处理
hrf例子
https://mmsegmentation.readthedocs.io/zh_CN/latest/dataset_prepare.html

(mmsegmentation) admin@bogon mmsegmentation %  python tools/convert_datasets/hrf.py /Users/admin/data/HRF/healthy.zip /Users/admin/data/HRF/healthy_manualsegm.zip /Users/admin/data/HRF/glaucoma.zip /Users/admin/data/HRF/glaucoma_manualsegm.zip /Users/admin/data/HRF/diabetic_retinopathy.zip /Users/admin/data/HRF/diabetic_retinopathy_manualsegm.zip -o /Users/admin/data/HRF/

(mmsegmentation) admin@bogon mmsegmentation % python tools/train.py configs/unet/fcn_unet_s5-d16_256x256_40k_hrf.py
2022-11-08 19:22:56,606 - mmseg - INFO - Multi-processing start method is `None`
2022-11-08 19:22:56,607 - mmseg - INFO - OpenCV num_threads is `8
Configured with: --prefix=/Users/admin/Downloads/Xcode.app/Contents/Developer/usr --with-gxx-include-dir=/Users/admin/Downloads/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/4.2.1
2022-11-08 19:22:56,685 - mmseg - INFO - Environment info:
------------------------------------------------------------
sys.platform: darwin
Python: 3.7.13 (default, Mar 28 2022, 07:24:34) [Clang 12.0.0 ]
CUDA available: False
GCC: Apple clang version 12.0.0 (clang-1200.0.32.2)
PyTorch: 1.5.0
PyTorch compiling details: PyTorch built with:
  - GCC 4.2
  - C++ Version: 201402
  - clang 9.1.0
  - Intel(R) Math Kernel Library Version 2019.0.5 Product Build 20190808 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v0.21.1 (Git Hash 7d2fd500bc78936d1d648ca713b901012f470dbc)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - Build settings: BLAS=MKL, BUILD_TYPE=Release, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -Wno-deprecated-declarations -DNDEBUG -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_INTERNAL_THREADPOOL_IMPL -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-typedef-redefinition -Wno-unknown-warning-option -Wno-unused-private-field -Wno-inconsistent-missing-override -Wno-aligned-allocation-unavailable -Wno-c++14-extensions -Wno-constexpr-not-const -Wno-missing-braces -Qunused-arguments -fcolor-diagnostics -faligned-new -fno-math-errno -fno-trapping-math -Werror=format -Wno-unused-private-field -Wno-missing-braces -Wno-c++14-extensions -Wno-constexpr-not-const, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=OFF, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=OFF, USE_STATIC_DISPATCH=OFF, 

TorchVision: 0.6.0
OpenCV: 4.6.0
MMCV: 1.5.0
MMCV Compiler: clang 12.0.0
MMCV CUDA Compiler: not available
MMSegmentation: 0.29.1+7b09967
------------------------------------------------------------

2022-11-08 19:22:56,686 - mmseg - INFO - Distributed training: False
2022-11-08 19:22:56,981 - mmseg - INFO - Config:
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
dataset_type = 'HRFDataset'
data_root = '/Users/admin/data/HRF'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (2336, 3504)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2336, 3504), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2336, 3504),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=40,
        dataset=dict(
            type='HRFDataset',
            data_root='/Users/admin/data/HRF',
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(2336, 3504),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(256, 256),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='HRFDataset',
        data_root='/Users/admin/data/HRF',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2336, 3504),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='HRFDataset',
        data_root='/Users/admin/data/HRF',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2336, 3504),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40)
checkpoint_config = dict(by_epoch=False, interval=40)
evaluation = dict(interval=40, metric='mDice', pre_eval=True)
work_dir = './work_dirs/fcn_unet_s5-d16_256x256_40k_hrf'
gpu_ids = [0]
auto_resume = False

2022-11-08 19:22:56,982 - mmseg - INFO - Set random seed to 901724302, deterministic: False
/Users/admin/opt/anaconda3/envs/mmsegmentation/lib/python3.7/site-packages/mmseg/models/decode_heads/decode_head.py:94: UserWarning: For binary segmentation, we suggest using`out_channels = 1` to define the outputchannels of segmentor, and use `threshold`to convert seg_logist into a predictionapplying a threshold
  warnings.warn('For binary segmentation, we suggest using'
/Users/admin/opt/anaconda3/envs/mmsegmentation/lib/python3.7/site-packages/mmseg/models/losses/cross_entropy_loss.py:236: UserWarning: Default ``avg_non_ignore`` is False, if you would like to ignore the certain label and average loss over non-ignore labels, which is the same with PyTorch official cross_entropy, set ``avg_non_ignore=True``.
  'Default ``avg_non_ignore`` is False, if you would like to '
2022-11-08 19:22:57,436 - mmseg - INFO - initialize UNet with init_cfg [{'type': 'Kaiming', 'layer': 'Conv2d'}, {'type': 'Constant', 'val': 1, 'layer': ['_BatchNorm', 'GroupNorm']}]
2022-11-08 19:22:57,668 - mmseg - INFO - initialize FCNHead with init_cfg {'type': 'Normal', 'std': 0.01, 'override': {'name': 'conv_seg'}}
2022-11-08 19:22:57,669 - mmseg - INFO - initialize FCNHead with init_cfg {'type': 'Normal', 'std': 0.01, 'override': {'name': 'conv_seg'}}
tools/train.py:207: UserWarning: SyncBN is only supported with DDP. To be compatible with DP, we convert SyncBN to BN. Please use dist_train.sh which can avoid this error.
  'SyncBN is only supported with DDP. To be compatible with DP, '
2022-11-08 19:22:57,674 - mmseg - INFO - EncoderDecoder(
  (backbone): UNet(
    (encoder): ModuleList(
      (0): Sequential(
        (0): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
      (1): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
      (2): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
      (3): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
      (4): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
    )
    (decoder): ModuleList(
      (0): UpConvBlock(
        (conv_block): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
        (upsample): InterpConv(
          (interp_upsample): Sequential(
            (0): Upsample()
            (1): ConvModule(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): _BatchNormXd(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
      (1): UpConvBlock(
        (conv_block): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
        (upsample): InterpConv(
          (interp_upsample): Sequential(
            (0): Upsample()
            (1): ConvModule(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): _BatchNormXd(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
      (2): UpConvBlock(
        (conv_block): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
        (upsample): InterpConv(
          (interp_upsample): Sequential(
            (0): Upsample()
            (1): ConvModule(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): _BatchNormXd(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
      (3): UpConvBlock(
        (conv_block): BasicConvBlock(
          (convs): Sequential(
            (0): ConvModule(
              (conv): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
            (1): ConvModule(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
        (upsample): InterpConv(
          (interp_upsample): Sequential(
            (0): Upsample()
            (1): ConvModule(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(inplace=True)
            )
          )
        )
      )
    )
  )
  init_cfg=[{'type': 'Kaiming', 'layer': 'Conv2d'}, {'type': 'Constant', 'val': 1, 'layer': ['_BatchNorm', 'GroupNorm']}]
  (decode_head): FCNHead(
    input_transform=None, ignore_index=255, align_corners=False
    (loss_decode): CrossEntropyLoss(avg_non_ignore=False)
    (conv_seg): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
    (dropout): Dropout2d(p=0.1, inplace=False)
    (convs): Sequential(
      (0): ConvModule(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): _BatchNormXd(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
    )
  )
  init_cfg={'type': 'Normal', 'std': 0.01, 'override': {'name': 'conv_seg'}}
  (auxiliary_head): FCNHead(
    input_transform=None, ignore_index=255, align_corners=False
    (loss_decode): CrossEntropyLoss(avg_non_ignore=False)
    (conv_seg): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
    (dropout): Dropout2d(p=0.1, inplace=False)
    (convs): Sequential(
      (0): ConvModule(
        (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): _BatchNormXd(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
    )
  )
  init_cfg={'type': 'Normal', 'std': 0.01, 'override': {'name': 'conv_seg'}}
)
2022-11-08 19:22:57,680 - mmseg - INFO - Loaded 15 images
2022-11-08 19:22:58,053 - mmseg - INFO - Loaded 30 images
2022-11-08 19:22:58,053 - mmseg - INFO - Start running, host: admin@bogon, work_dir: /Users/admin/data/test_project/mmsegmentation/work_dirs/fcn_unet_s5-d16_256x256_40k_hrf





torch.expand(-1, -1)的理解
在expand中的-1表示取当前所在维度的尺寸，也就是表示当前维度不变。
在代码中 一般用这方法解决不想手动计算维度的时候

import torch
x = torch.Tensor([[1], [2], [3]])
x0 = x.size(0)  # 取x第一维的尺寸，x0 = 3
x1 = x.expand(-1, 2)
x2 = x.expand(3, 2)

输出
x0 =  3
x1 =  tensor([[1., 1.],
        [2., 2.],
        [3., 3.]])
x2 =  tensor([[1., 1.],
        [2., 2.],
        [3., 3.]])




(mmsegmentation) admin@bogon mmsegmentation % python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug.py


model
mmseg/models/segmentors/encoder_decoder.py-->EncoderDecoder
1、pretrain
2、backbone
3、_init_decode_head
4、_init_auxiliary_head
5、train_cfg


backbone 
mmseg/models/backbones/resnet.py-->ResNetV1c






beit 图片的尺寸会有影响
(mmsegmentation) admin@bogon mmsegmentation % python tools/train.py configs/beit/upernet_beit-base_8x2_640x640_160k_ade20k.py

EncoderDecoder
1、backbone -->beit
    patch
    layer-->BEiTTransformerEncoderLayer
        self.attn
        self.ffn(cfg)
            mmseg/models/backbones/vit.py
            ffn_cfg.update(
                dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,#全联接层
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))



decode_head
mmseg/models/decode_heads/uper_head.py
UPerHead--(multiple_select)
self._forward_feature(inputs)
    decode_head : UPerHead_self_paramer {'in_channels': [768, 768, 768, 768], 'in_index': [0, 1, 2, 3], 'channels': 768, 'dropout_ratio': 0.1, 'num_classes': 150, 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 'align_corners': False, 'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}}
    1、self._transform_inputs
    2、build_laterals + psp_modules
    3、fpn_outs--resize到同一个尺寸，cat
    4、fpn_bottleneck(fpn_outs)
self.cls_seg(output)
    self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)  self.out_channels即class
    

auxiliary_head(可以是列表)
self._init_auxiliary_head(auxiliary_head) fcn全联接





beit_patch_input : x torch.Size([2, 3, 640, 640])
beit_patch_adap : x torch.Size([2, 3, 640, 640])
beit_patch_projection : x torch.Size([2, 768, 40, 40]) cfg dim 768
beit_input : inputs torch.Size([2, 3, 640, 640]) ,path_out_x torch.Size([2, 1600, 768]) , path_out_hw (40, 40)
beit_expand : x torch.Size([2, 1601, 768])
beit_out : [torch.Size([2, 768, 40, 40]), torch.Size([2, 768, 40, 40]), torch.Size([2, 768, 40, 40]), torch.Size([2, 768, 40, 40])]

encode_decode : img torch.Size([2, 3, 640, 640]) 
decode_head : laterals [torch.Size([2, 768, 160, 160]), torch.Size([2, 768, 80, 80]), torch.Size([2, 768, 40, 40])]
decode_head_laterals_psp : [torch.Size([2, 768, 20, 20]), torch.Size([2, 768, 20, 20]), torch.Size([2, 768, 20, 20]), torch.Size([2, 768, 20, 20]), torch.Size([2, 768, 20, 20])]
decode_head_laterals_psp_cat : torch.Size([2, 3840, 20, 20])
decode_head_psp_out : torch.Size([2, 768, 20, 20])



beit_



























mae
(mmsegmentation) admin@bogon mmsegmentation % python tools/train.py configs/mae/upernet_mae-base_fp16_8x2_512x512_160k_ade20k.py







batchsize=1的情况
Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])
https://blog.csdn.net/hjxu2016/article/details/121075723





Logger模块 log_level=logging.INFO print输出改为Logger输出，记录不同级别的信息
Log_debug = get_root_logger('./debug.log',log_level=logging.ERROR)




cityscapes数据数
Cityscapes数据集的深度完整解析
https://blog.csdn.net/MVandCV/article/details/115331719

(mmsegmentation) admin@bogon mmsegmentation % python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py

mmseg/datasets/custom.py
init
self.file_client = mmcv.FileClient.infer_client(self.file_client_args)  # self.file_client_args=dict(backend='disk')
    _backends = {
        'disk': HardDiskBackend,
        'ceph': CephBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
        'petrel': PetrelBackend,
        'http': HTTPBackend,
    }
for img in self.file_client.list_dir_or_file(
        dir_path=img_dir,
        list_dir=False,
        suffix=img_suffix,
        recursive=True)
    self.client (_instance.client = cls._backends[backend](**kwargs)) 返回不同类型的文件客户端，list_dir_or_file继承HardDiskBackend的方法
    返回 yield from self.client.list_dir_or_file(dir_path, list_dir, list_file,suffix, recursive)

返回img_info信息
self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)
 
 
 
 
 def __new__(cls, backend=None, prefix=None, **kwargs):
 __new__() 是一种负责创建类实例的静态方法，它无需使用 staticmethod 装饰器修饰，且该方法会优先 __init__() 初始化方法被调用
 
路径迭代器 (list_dir=False)
def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                     recursive):
   for entry in os.scandir(dir_path):
       if not entry.name.startswith('.') and entry.is_file():
           rel_path = osp.relpath(entry.path, root)
           if (suffix is None
                   or rel_path.endswith(suffix)) and list_file:
               yield rel_path
       elif osp.isdir(entry.path): 判断文件夹
           if list_dir: #这里不会走
               rel_dir = osp.relpath(entry.path, root)
               yield rel_dir
           if recursive:
               yield from _list_dir_or_file(entry.path, list_dir,
                                            list_file, suffix,
                                            recursive)
 
 


2022-11-17 13:59:09,362 - mmseg - INFO - CustomDataset : paramer {'pipeline': Compose(
    LoadImageFromFile(to_float32=False,color_type='color',imdecode_backend='cv2')
    MultiScaleFlipAug(transforms=Compose(
    Resize(img_scale=None, multiscale_mode=range, ratio_range=None, keep_ratio=True)
    RandomFlip(prob=None)
    Normalize(mean=[123.675 116.28  103.53 ], std=[58.395 57.12  57.375], to_rgb=True)
    ImageToTensor(keys=['img'])
    Collect(keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'))
), img_scale=[(2048, 1024)], flip=False)flip_direction=['horizontal']
), 'img_dir': '/Users/admin/Downloads/cityscapes/val', 'img_suffix': '_leftImg8bit.png', 'ann_dir': '/Users/admin/Downloads/cityscapes/gtFine/val', 'seg_map_suffix': '_gtFine_labelTrainIds.png', 'split': None, 'data_root': '/Users/admin/Downloads/cityscapes/', 'test_mode': True, 'ignore_index': 255, 'reduce_zero_label': False, 'label_map': None, 'custom_classes': False, 'CLASSES': ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'), 'PALETTE': [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], 'gt_seg_map_loader': LoadAnnotations(reduce_zero_label=False,imdecode_backend='pillow'), 'file_client_args': {'backend': 'disk'}, 'file_client': <mmcv.fileio.file_client.FileClient object at 0x7f988a342950>, 'img_infos': [{'filename': 'frankfurt/frankfurt_000000_000294_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_000576_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_000576_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_001016_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_001016_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_001236_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_001236_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_001751_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_001751_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_002196_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_002196_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_002963_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_002963_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_003025_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_003025_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_003357_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_003357_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_003920_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_003920_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_004617_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_004617_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_005543_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_005543_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_005898_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_005898_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_006589_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_006589_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_007365_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_007365_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_008206_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_008206_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_008451_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_008451_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_009291_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_009291_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_009561_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_009561_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_009688_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_009688_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_009969_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_009969_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_010351_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_010351_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_010763_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_010763_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_011007_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_011007_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_011074_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_011074_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_011461_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_011461_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_011810_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_011810_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_012009_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_012009_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_012121_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_012121_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_012868_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_012868_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_013067_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_013067_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_013240_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_013240_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_013382_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_013382_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_013942_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_013942_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_014480_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_014480_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_015389_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_015389_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_015676_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_015676_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_016005_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_016005_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_016286_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_016286_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_017228_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_017228_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_017476_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_017476_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_018797_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_018797_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_019607_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_019607_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_020215_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_020215_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_020321_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_020321_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_020880_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_020880_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_021667_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_021667_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_021879_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_021879_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_022254_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_022254_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000000_022797_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000000_022797_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_000538_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_000538_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_001464_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_001464_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_002512_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_002512_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_002646_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_002646_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_002759_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_002759_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_003056_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_003056_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_003588_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_003588_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_004327_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_004327_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_004736_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_004736_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_004859_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_004859_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_005184_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_005184_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_005410_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_005410_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_005703_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_005703_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_005898_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_005898_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_007285_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_007285_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_007407_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_007407_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_007622_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_007622_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_007857_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_007857_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_007973_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_007973_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_008200_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_008200_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_008688_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_008688_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_009058_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_009058_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_009504_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_009504_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_009854_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_009854_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_010156_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_010156_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_010444_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_010444_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_010600_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_010600_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_010830_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_010830_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_011162_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_011162_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_011715_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_011715_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_011835_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_011835_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_012038_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_012038_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_012519_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_012519_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_012699_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_012699_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_012738_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_012738_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_012870_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_012870_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_013016_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_013016_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_013496_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_013496_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_013710_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_013710_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_014221_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_014221_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_014406_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_014406_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_014565_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_014565_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_014741_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_014741_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_015091_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_015091_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_015328_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_015328_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_015768_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_015768_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_016029_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_016029_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_016273_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_016273_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_016462_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_016462_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_017101_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_017101_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_017459_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_017459_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_017842_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_017842_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_018113_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_018113_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_019698_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_019698_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_019854_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_019854_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_019969_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_019969_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_020046_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_020046_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_020287_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_020287_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_020693_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_020693_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_021406_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_021406_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_021825_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_021825_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_023235_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_023235_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_023369_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_023369_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_023769_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_023769_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_024927_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_024927_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_025512_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_025512_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_025713_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_025713_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_025921_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_025921_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_027325_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_027325_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_028232_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_028232_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_028335_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_028335_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_028590_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_028590_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_028854_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_028854_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_029086_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_029086_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_029236_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_029236_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_029600_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_029600_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_030067_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_030067_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_030310_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_030310_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_030669_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_030669_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_031266_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_031266_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_031416_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_031416_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_032018_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_032018_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_032556_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_032556_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_032711_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_032711_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_032942_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_032942_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_033655_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_033655_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_034047_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_034047_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_034816_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_034816_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_035144_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_035144_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_035864_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_035864_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_037705_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_037705_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_038245_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_038245_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_038418_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_038418_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_038645_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_038645_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_038844_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_038844_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_039895_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_039895_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_040575_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_040575_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_040732_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_040732_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_041074_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_041074_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_041354_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_041354_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_041517_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_041517_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_041664_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_041664_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_042098_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_042098_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_042384_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_042384_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_042733_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_042733_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_043395_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_043395_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_043564_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_043564_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_044227_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_044227_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_044413_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_044413_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_044525_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_044525_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_044658_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_044658_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_044787_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_044787_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_046126_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_046126_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_046272_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_046272_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_046504_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_046504_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_046779_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_046779_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_047178_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_047178_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_047552_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_047552_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_048196_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_048196_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_048355_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_048355_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_048654_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_048654_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_049078_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_049078_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_049209_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_049209_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_049298_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_049298_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_049698_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_049698_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_049770_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_049770_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_050149_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_050149_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_050686_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_050686_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_051516_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_051516_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_051737_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_051737_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_051807_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_051807_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_052120_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_052120_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_052594_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_052594_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_053102_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_053102_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_054077_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_054077_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_054219_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_054219_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_054415_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_054415_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_054640_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_054640_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_054884_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_054884_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_055062_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_055062_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_055172_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_055172_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_055306_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_055306_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_055387_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_055387_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_055538_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_055538_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_055603_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_055603_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_055709_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_055709_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_056580_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_056580_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_057181_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_057181_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_057478_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_057478_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_057954_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_057954_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_058057_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_058057_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_058176_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_058176_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_058504_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_058504_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_058914_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_058914_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_059119_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_059119_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_059642_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_059642_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_059789_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_059789_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_060135_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_060135_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_060422_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_060422_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_060545_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_060545_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_060906_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_060906_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_061682_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_061682_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_061763_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_061763_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_062016_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_062016_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_062250_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_062250_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_062396_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_062396_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_062509_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_062509_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_062653_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_062653_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_062793_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_062793_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_063045_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_063045_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_064130_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_064130_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_064305_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_064305_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_064651_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_064651_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_064798_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_064798_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_064925_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_064925_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_065160_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_065160_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_065617_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_065617_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_065850_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_065850_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_066092_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_066092_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_066438_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_066438_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_066574_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_066574_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_066832_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_066832_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_067092_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_067092_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_067178_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_067178_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_067295_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_067295_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_067474_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_067474_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_067735_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_067735_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_068063_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_068063_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_068208_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_068208_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_068682_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_068682_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_068772_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_068772_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_069633_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_069633_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_070099_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_070099_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_071288_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_071288_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_071781_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_071781_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_072155_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_072155_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_072295_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_072295_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_073088_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_073088_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_073243_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_073243_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_073464_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_073464_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_073911_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_073911_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_075296_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_075296_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_075984_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_075984_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_076502_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_076502_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_077092_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_077092_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_077233_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_077233_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_077434_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_077434_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_078803_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_078803_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_079206_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_079206_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_080091_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_080091_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_080391_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_080391_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_080830_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_080830_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_082087_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_082087_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_082466_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_082466_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_083029_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_083029_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_083199_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_083199_gtFine_labelTrainIds.png'}}, {'filename': 'frankfurt/frankfurt_000001_083852_leftImg8bit.png', 'ann': {'seg_map': 'frankfurt/frankfurt_000001_083852_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000000_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000000_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000001_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000001_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000002_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000002_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000003_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000003_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000004_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000004_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000005_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000005_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000006_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000006_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000007_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000007_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000008_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000008_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000009_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000009_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000010_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000010_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000011_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000011_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000012_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000012_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000013_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000013_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000014_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000014_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000015_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000015_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000016_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000016_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000017_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000017_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000018_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000018_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000019_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000019_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000020_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000020_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000021_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000021_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000022_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000022_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000023_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000023_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000024_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000024_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000025_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000025_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000026_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000026_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000027_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000027_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000028_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000028_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000029_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000029_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000030_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000030_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000031_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000031_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000032_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000032_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000033_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000033_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000034_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000034_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000035_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000035_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000036_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000036_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000037_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000037_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000038_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000038_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000039_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000039_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000040_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000040_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000041_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000041_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000042_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000042_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000043_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000043_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000044_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000044_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000045_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000045_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000046_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000046_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000047_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000047_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000048_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000048_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000049_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000049_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000050_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000050_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000051_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000051_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000052_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000052_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000053_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000053_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000054_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000054_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000055_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000055_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000056_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000056_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000057_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000057_000019_gtFine_labelTrainIds.png'}}, {'filename': 'lindau/lindau_000058_000019_leftImg8bit.png', 'ann': {'seg_map': 'lindau/lindau_000058_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000000_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000000_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000001_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000001_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000002_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000002_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000003_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000003_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000004_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000004_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000005_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000005_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000006_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000006_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000007_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000007_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000008_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000008_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000009_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000009_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000010_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000010_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000011_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000011_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000012_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000012_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000013_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000013_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000014_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000014_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000015_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000015_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000016_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000016_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000017_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000017_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000018_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000018_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000019_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000019_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000020_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000020_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000021_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000021_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000022_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000022_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000023_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000023_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000024_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000024_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000025_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000025_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000026_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000026_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000027_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000027_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000028_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000028_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000029_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000029_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000030_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000030_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000031_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000031_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000032_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000032_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000033_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000033_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000034_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000034_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000035_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000035_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000036_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000036_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000037_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000037_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000038_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000038_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000039_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000039_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000040_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000040_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000041_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000041_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000042_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000042_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000043_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000043_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000044_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000044_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000045_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000045_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000046_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000046_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000047_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000047_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000048_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000048_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000049_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000049_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000050_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000050_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000051_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000051_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000052_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000052_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000053_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000053_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000054_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000054_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000055_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000055_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000056_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000056_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000057_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000057_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000058_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000058_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000059_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000059_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000060_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000060_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000061_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000061_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000062_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000062_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000063_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000063_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000064_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000064_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000065_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000065_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000066_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000066_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000067_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000067_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000068_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000068_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000069_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000069_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000070_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000070_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000071_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000071_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000072_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000072_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000073_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000073_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000074_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000074_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000075_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000075_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000076_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000076_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000077_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000077_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000078_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000078_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000079_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000079_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000080_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000080_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000081_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000081_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000082_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000082_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000083_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000083_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000084_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000084_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000085_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000085_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000086_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000086_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000087_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000087_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000088_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000088_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000089_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000089_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000090_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000090_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000091_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000091_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000092_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000092_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000093_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000093_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000094_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000094_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000095_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000095_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000096_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000096_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000097_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000097_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000098_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000098_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000099_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000099_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000100_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000100_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000101_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000101_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000102_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000102_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000103_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000103_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000104_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000104_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000105_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000105_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000106_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000106_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000107_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000107_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000108_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000108_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000109_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000109_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000110_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000110_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000111_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000111_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000112_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000112_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000113_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000113_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000114_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000114_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000115_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000115_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000116_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000116_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000117_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000117_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000118_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000118_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000119_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000119_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000120_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000120_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000121_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000121_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000122_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000122_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000123_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000123_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000124_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000124_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000125_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000125_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000126_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000126_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000127_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000127_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000128_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000128_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000129_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000129_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000130_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000130_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000131_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000131_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000132_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000132_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000133_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000133_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000134_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000134_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000135_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000135_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000136_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000136_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000137_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000137_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000138_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000138_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000139_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000139_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000140_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000140_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000141_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000141_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000142_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000142_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000143_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000143_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000144_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000144_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000145_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000145_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000146_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000146_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000147_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000147_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000148_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000148_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000149_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000149_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000150_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000150_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000151_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000151_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000152_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000152_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000153_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000153_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000154_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000154_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000155_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000155_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000156_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000156_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000157_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000157_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000158_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000158_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000159_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000159_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000160_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000160_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000161_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000161_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000162_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000162_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000163_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000163_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000164_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000164_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000165_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000165_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000166_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000166_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000167_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000167_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000168_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000168_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000169_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000169_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000170_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000170_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000171_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000171_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000172_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000172_000019_gtFine_labelTrainIds.png'}}, {'filename': 'munster/munster_000173_000019_leftImg8bit.png', 'ann': {'seg_map': 'munster/munster_000173_000019_gtFine_labelTrainIds.png'}}]} 







ADE20k数据集
* . jpg：RGB图像。
* _seg.png：对象分割掩码。此图像包含有关对象类分割掩码的信息，还将每个类分隔为实例。通道R和G编码对象类掩码。通道B对实例对象掩码进行编码。loadAde20K的函数。m提取两个掩模。
* _seg_parts_N.png：零件分割掩码，其中N是一个数字(1,2,3，…)，表示零件层次结构中的级别。部件被组织在一个树中，其中对象由部件组成，部件也可以由部件组成，部件的部件也可以有部件。level N表示部件树中的深度。级别N=1对应于对象的各个部分。所有的部件分割都具有与对象分割掩码相同的编码，类在RG通道中编码，实例在B通道中编码。使用loadAde20K函数。提取部分分割掩码，并将同一类的实例分离。
* _.txt：描述每个图像(描述对象和部件)内容的文本文件。此信息与其他文件是冗余的。但另外还包含有关对象属性的信息。loadAde20K的函数。m还解析这个文件的内容。文本文件中的每一行包含:列1=实例号，列2=部件级别(对象为0)，列3=遮挡(true为1)，列4=类名(使用wordnet解析)，列5=原始名称(可能提供更详细的分类)，列6=逗号分隔的属性列表。





beit : self.layers ModuleList(
  (0): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (1): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (2): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (3): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (4): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (5): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (6): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (7): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (8): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (9): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (10): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
  (11): BEiTTransformerEncoderLayer(
    (ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (attn): BEiTAttention(
      (qkv): Linear(in_features=768, out_features=2304, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=768, out_features=768, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (ffn): FFN(
      (activate): GELU()
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=3072, out_features=768, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (drop_path): DropPath()
  )
) , self.out_indices (3, 5, 7, 11)




train.py
train_segmentor(model,datasets,cfg,distributed=distributed,validate=(not args.no_validate),timestamp=timestamp,meta=meta)
    mmseg/apis/train.py
    train_segmentor -- >构建runner
    runner = build_runner(cfg.runner,default_args=dict(model=model,batch_processor=None,optimizer=optimizer,work_dir=cfg.work_dir,logger=logger,meta=meta))
    runner.run(data_loaders, cfg.workflow)
        /Users/admin/opt/anaconda3/envs/mmsegmentation/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py-->run
        epoch_runner = getattr(self, mode)
        epoch_runner(data_loaders[i], **kwargs)-->如何调用train
            outputs = self.model.train_step(data_batch, self.optimizer,**kwargs)
            mmseg/models/segmentors/base.py-->train_step
            losses = self(**data_batch)  **data_batch字典
                mmseg/models/segmentors/encoder_decoder.py
                forward_train(self, img, img_metas, gt_semantic_seg)
            
            
        




语义分割轻量网络
1、icnet
(mmsegmentation) admin@AdmindeMacBook-Pro-47 mmsegmentation % python tools/train.py configs/icnet/icnet_r50-d8_832x832_80k_cityscapes.py

backbone
conv_sub1 通道扩充到middle_channels，输出out_channels[0]。三次降采stride=2，缩小8倍
ResNetV1c (初期的resnet模型，stem与conv1的功能相同)
_make_stem_layer-->(deep_stem=True,avg_down=False) 
layer1
layer2
layer3
layer4

psp_module (Pyramid Pooling Module)


neck
cff集联，通道数是相同的，大小可能不同。
1、resize到相同大小。
2、conv_low空洞卷积带空洞率，conv_high两个卷积处理，维持输入尺寸。
3、合并
4、relu激活

第二、四层做cff24
第一，cff24做cff 返回cff12、调整尺寸后的cff24


decode_head
fcn




2、cgnet
(mmsegmentation) admin@AdmindeMacBook-Pro-47 mmsegmentation % python tools/train.py configs/cgnet/cgnet_680x680_60k_cityscapes.py


inject_2x=InputInjection(1)-->self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
inject_4x=InputInjection(2) 2倍
stage0
stem 第一个stride=2其他stride都为1，进入的x降采一半儿
cat(x,inject_2x) 保留了原始的输入数据的上下文信息

stage1（x已经融入了stage0的信息）
ContextGuidedBlock 4部分
self.conv1x1 kernel大小
self.f_loc常规卷积局部提取
self.f_sur空洞卷积上下文提取
cat([loc, sur], 1)拼接
self.bottleneck 通道数将为原来一半（downsample只有i=0的时候为true）















