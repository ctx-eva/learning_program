# Summary of YOLOv8

This is a summary of yolov8 learning process. It contains some tasks implementation process, model structrue, code annotation.

The original repository link is: https://github.com/ultralytics/ultralytics

The summary is based on version 8.0.49

## YOLOv8文件结构以及文件功能介绍

```
|-- __init__.py
|-- datasets/                          # 多种类型数据格式文件配置
|-- hub
|   |-- __init__.py
|   |-- auth.py
|   |-- session.py
|   `-- utils.py
|-- models
|   |-- v3/                            # yolov3网络的配置文件
|   |-- v5/                            # yolov5网络的配置文件
|   `-- v8/
|       |-- cls/                       # yolov8分类网络的配置文件
|       |-- seg/                       # yolov8实例分割网络的配置文件
|       |-- yolov8.yaml                # yolov8目标检测网络的配置文件
|-- nn
|   |-- __init__.py
|   |-- autobackend.py                 # 提供了多种类型的模型加载方式,并且为各种类型模型的forward运算提供接入点
|   |-- autoshape.py
|   |-- modules.py                     # yolo项目包含的基本模型单元结构
|   `-- tasks.py                       # 提供模型加载和模型forward运算的基类BaseModel,并为classify,Detect,segment分别实现了对应的SegmentationModel类,
|                                      #     DetectionModel类和ClassificationModel类,为autobackend提供模型文件类型是torch的.pt文件时的加载实现.
|                                      #     是模型进行forward运算的实现位置.并且在BaseModel的fuse函数中做了对conv+bn的参数融合.
|-- tracker/                           # 目标跟踪的部分在../target_tracking部分完成后在更新
|
`-- yolo
    |-- __init__.py
    |-- cfg                            # 网络的默认超参数配置位置
    |   |-- __init__.py
    |   `-- default.yaml
    |-- data
    |   |-- __init__.py
    |   |-- augment.py                 # 1.为推断过程提供LetterBox类处理输入图像的尺寸,
    |   |                              # 2.为检测和分割train,val任务YOLODataset类提供了v8_transforms数据扩增变换函数,其中包含多种形式的图像变换类,用RandomPerspective类为对应变换的提供相应目标坐标变换.
    |   |                              #     为classify任务ClassificationDataset类提供了classify_transforms,classify_albumentations的数据扩增函数
    |   |-- base.py                    #
    |   |-- build.py                   # 为train,val,inference过程提供合适的数据加载函数dataloader
    |   |-- dataloaders                
    |   |   |-- __init__.py            
    |   |   |-- stream_loaders.py      # 为推断过程提供多种形式的图片读入方式,按要求对图片进行变换,并调整图片尺寸以适应网络
    |   |   |-- v5augmentations.py     
    |   |   `-- v5loader.py            
    |   |-- dataset.py                 # dataloader的主体实现检测和分割过程为YOLODataset类,分类过程为ClassificationDataset类
    |   |-- dataset_wrappers.py        
    |   `-- utils.py                   # 1.为augment提供mask计算 2.为train,val过程验证数据是否存在
    |-- engine
    |   |-- __init__.py
    |   |-- exporter.py                # 实现多种类型的模型文件导出
    |   |-- model.py                   # YOLO类:创建模型实例,初始化模型,加载模型权重的位置,也是所有任务(train,val,predict,export)的入口
    |   |-- predictor.py               # BasePredictor类:是DetectionPredictor和ClassificationPredictor的基类,主要提供stream_inference函数实现预测过程,
    |   |                              #     主要包含输入图像加载,创建autobackend 对象实现模型加载,调用autobackend forward进行模型运算,
    |   |                              #     在该类实现了对预测过程的结果进行保存和展示
    |   |-- results.py                 # 对pred任务输出进行规范实例化,并提供结果多种形式的变换
    |   |-- trainer.py                 # BasePredictor类:作为DetectionTrainer和ClassificationTrainer的基类,主要为训练过程提供优化器optimizer,训练中的模型加载和保存,训练中断时训练恢复.
    |   |                              #     是train每个epoch过程的具体实现位置,包括初始化时的optimizer,dataloader,validator的装载,model前向反向,loss的计算,过程中Validater的调用
    |   `-- validator.py               # BaseValidator类:作为SegmentationValidator类和ClassificationValidator类的基类.提供单独进行model.val()时的接入口,是val每次过程的具体实现位置,是最终的val指标统计位置
    |-- utils
    |   |-- __init__.py
    |   |-- autobatch.py               # 当输入batch=-1时,根据cuda memory计算能够吞吐的batch大小
    |   |-- benchmarks.py
    |   |-- callbacks
    |   |   |-- __init__.py
    |   |   |-- base.py
    |   |   |-- clearml.py
    |   |   |-- comet.py
    |   |   |-- hub.py
    |   |   `-- tensorboard.py
    |   |-- checks.py                  # 主要任务: 1.项目的运行环境检测. 2.对batch输入图像的适当图像大小计算 3.模型文件和模型配置文件是否存在记忆是否下载的检测
    |   |-- dist.py                    # 分布式运算的网络连接
    |   |-- downloads.py               # 在各过程中若找不到模型文件,实现对应的模型文件的下载
    |   |-- files.py                   # 主要功能是在进行各类文件保存时,递归创建对应的文件路径
    |   |-- instance.py                # 对象实例的Instances类和bbox类,实现了Instances对象的变换,以及bbox类中所需的计算
    |   |-- loss.py                    # 损失函数的实现,包含VFL,在目标框的分类和回归中实现了DFL. 损失函数的详细解释请见(../../../loss_function)
    |   |-- metrics.py                 # 主要任务: 1.框iou的计算 2.提供Focal loss计算, 3.训练和验证过程的指标运算及指标作图
    |   |-- ops.py                     # 算子文件 主要任务:1.网络推断结果的NMS运算实现 2.对输出结果向原图的操作,边界限制,大小恢复等 3.segment对象向box对象转换
    |   |-- plotting.py                # Annotator类为pred过程作图的类实现,并为train,val过程提供作图函数
    |   |-- tal.py                     # TAL:Task Alignment Learning 训练过程中的目标框(正样本)分配策略
    |   `-- torch_utils.py             # 主要任务:1.DP or DDP 分布式训练的实现位置 2.tasks.py中conv+bn的参数融合的具体实现位置.
    `-- v8
        |-- __init__.py
        |-- classify
        |   |-- __init__.py
        |   |-- predict.py             # ClassificationPredictor类:重载了BasePredictor类的图像数据类型预处理,模型结果后处理,结果作图环节.是model函数的classify任务接入点
        |   |-- train.py               # ClassificationTrainer类:重载了DetectionTrainer类中的dataloader(ClassificationDataset),validator(ClassificationValidator),criterion(BCE)
        |   `-- val.py                 # ClassificationValidator类,重载了BaseValidator中的大部分函数,包含自身的数据加载前后处理,nms计算,作图和val指标计算
        |-- detect
        |   |-- __init__.py
        |   |-- predict.py             # DetectionPredictor类:重载了BasePredictor类的图像数据类型预处理,模型结果后处理,结果作图环节. 是model函数的detect任务接入点, 
        |   |                          #     并作为segment任务 SegmentationPredictor的父类
        |   |-- train.py               # 1.DetectionTrainer类:重载了BasePredictor类中的作图函数,get_model函数加载DetectionModel类作为train中forward过程,
        |   |                          #     get_validator函数加载DetectionValidator类作为train的验证过程,criterion函数将Loss类作为train的目标函数.
        |   |                          # 2.Loss类: preprocess函数排除batch中没有gt_box的图像, bbox_decode将对应reg_box分支的输出转化为softmax形式对应DFL的计算.
        |   |                          #     Loss计算过程使用TAL筛选正样本,并将reg_box除以stride_tensor进行归一化,用bce计算cls_loss,DFL计算dfl_loss,CIOU计算box_loss
        |   `-- val.py                 # DetectionValidator类,重载了BaseValidator中的大部分函数,包含自身的数据加载前后处理,nms计算,作图和val指标计算
        `-- segment
            |-- __init__.py
            |-- predict.py             # SegmentationPredictor类:重载了DetectionPredictor类的模型结果后处理和结果作图环节.并作为model函数的segment任务接入点
            |-- train.py               # 1.SegmentationTrainer类:重载了DetectionTrainer类中的作图环节,get_model函数加载SegmentationModel类作为train中forward过程,
            |                          #     get_validator函数加载SegmentationValidator类作为train的验证过程,criterion函数将SegLoss类作为train的目标函数.
            |                          # 2.SegLoss类:基类为train中Loss类,重载了loss的计算过程,增加了各层mask上各点的BCE作为mask_loss
            `-- val.py                 # SegmentationValidator类,重载了DetectionValidator中的大部分函数,包含自身的数据加载前后处理,nms计算,作图和val指标计算
```
