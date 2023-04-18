# Summary of YOLOv6

This is a summary of yolov8 learning process. It contains some tasks implementation process, model structrue, code annotation.

The original repository link is: https://github.com/meituan/YOLOv6

Implementation of paper:
- [YOLOv6 v3.0: A Full-Scale Reloading](https://arxiv.org/abs/2301.05586)
- [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976)

## YOLOv6文件结构以及文件功能介绍

```
.
|
|-- assets                                  # 量化实验的权重默认文件夹
|   |-- yolov6s_v2_reopt.pt
|   `-- yolov6s_v2_scale.pt
|-- configs
|   |-- base/                               # anchor base yolov6网络的配置文件
|   |-- experiment/                         # 包含评估参数的配置文件
|   |-- qarepvgg/                           # 使用QARepVGGBlock作为基本元件构建的anchor base yolov6网络的配置文件
|   |-- repopt/                             # 量化实验时所使用的yolov6网络的配置文件
|   |-- yolov6(?).py                        # anchor free yolov6网络的配置文件
|   `-- yolov6(?)6.py                       # 增加一层feature map结构的anchor free yolov6网络的配置文件
|-- data/                                   # 多种类型数据格式文件配置
|   `-- images/                             # 测试图像文件夹
|-- deploy
|   |-- ONNX
|   |   |-- OpenCV
|   |   |   |-- coco.names
|   |   |   |-- sample.jpg
|   |   |   |-- yolo.py                     # 加载ONNX模型对image数据进行推断并展示
|   |   |   |-- yolo_video.py               # 加载ONNX模型对video数据进行推断并展示
|   |   |   |-- yolov5
|   |   |   |   |-- CMakeLists.txt
|   |   |   |   `-- yolov5.cpp
|   |   |   |-- yolov6
|   |   |   |   |-- CMakeLists.txt
|   |   |   |   `-- yolov6.cpp
|   |   |   |-- yolox
|   |   |   |   |-- CMakeLists.txt
|   |   |   |   `-- yolox.cpp
|   |   |   `-- yolox.py
|   |   |-- eval_trt.py
|   |   `-- export_onnx.py                  # 加载FP32 model,进行RepVgg旁路结构向conv变换的deploy,按是否end2end,simplify,将pt模型转换为ONNX模型
|   |-- OpenVINO
|   |   `-- export_openvino.py              # 加载FP32 model,进行RepVgg旁路结构向conv变换的deploy,将pt模型转换为ONNX模型和OpenVINO模型
|   `-- TensorRT
|       |-- CMakeLists.txt
|       |-- Processor.py
|       |-- calibrator.py
|       |-- eval_yolo_trt.py
|       |-- logging.h
|       |-- onnx_to_trt.py
|       |-- visualize.py
|       `-- yolov6.cpp
|-- tools
|   |-- partial_quantization                # 项目中独立的ptq过程
|   |   |-- eval.py                         # 调用 Evaler.eval_model进行evaluation结果统计
|   |   |-- eval.yaml                       # partial量化模型的配置文件
|   |   |-- partial_quant.py                # 读取量化模型和sensitivity-layers,按配置生成对应的partial量化模型,调用eval统计模型性能,最终输出成partial量化的onnx模型
|   |   |-- ptq.py                          # 1.load_ptq和do_ptq分别进行量化模型加载和执行量化操作
|   |   |                                   #     2.使用nvidia pytorch-quantization库对conv,maxpool层进行数据校准和量化
|   |   |                                   #     3.partial_quantization可以有选择的跳过某些层并对剩下可以量化的层进行量化
|   |   |-- sensitivity_analyse.py          # 对整个模型量化,以及各个layer层单独量化,比较互相之间对于mAP的影响,输出sensitivity-layers
|   |   `-- utils.py                        # 量化模型的加载保存参数状态修改
|   |-- qat
|   |   |-- onnx_utils.py                   
|   |   |-- qat_export.py                   # 输出量化模型到ONNX模型
|   |   `-- qat_utils.py                    # 为Trainer类提供qat_init_model_manu,skip_sensitive_layers加载量化模型.
|   |-- quantization
|   |   |-- ppq
|   |   |   |-- ProgramEntrance.py
|   |   |   `-- write_qparams_onnx2trt.py
|   |   `-- tensorrt
|   |       |-- post_training
|   |       |   |-- Calibrator.py
|   |       |   |-- onnx_to_tensorrt.py
|   |       |   `-- quant.sh
|   |       `-- training_aware
|   |           `-- QAT_quantizer.py
|   |-- eval.py                             # evaluation过程调用的主函数入口,进行参数解析,创建Evaler类的实例进行校验
|   |-- infer.py                            # inferer过程调用的主函数入口,进行参数解析,创建Inferer类的实例进行推断
|   `-- train.py                            # train过程调用的主函数入口,进行参数解析,并检验权重和配置文件的路径存在判断是否要下载.创建Trainer类的实例进行训练
|-- weights                                 # 权重文件夹
|   `-- yolov6n.pt
|-- yolov6
|   |-- __init__.py
|   |-- assigners                           # 训练过程中的正样本分配策略 
|   |   |-- __init__.py
|   |   |-- anchor_generator.py             # 计算feature map上每个位置的anchor偏移量,缩放比例
|   |   |-- assigner_utils.py               # 为assigner提供框选择计算,gt-pred iou,为ATSSassigner提供gt-anchor center距离计算
|   |   |-- atss_assigner.py                # 训练过程中的目标框(正样本)分配策略ATSS: Adaptive Training Sample Selection Assigner
|   |   |-- iou2d_calculator.py             # 为assigner提供gt-anchor iou
|   |   `-- tal_assigner.py                 # 训练过程中的目标框(正样本)分配策略TAL:Task Alignment Learning 
|   |-- core
|   |   |-- engine.py                       # Trainer类:1.__init__中实例化dataloader,optimizer,加载模型在qat时加载教师模型,在qat时执行模型量化.
|   |   |                                   #     2.train_before_loop实例化损失函数 3.train_after_loop清空显存
|   |   |                                   #     4.train_in_loop调用prepare_for_steps,train_in_steps进行网络训练,调用eval_and_save进行结果评估和checkpoint保存
|   |   |                                   #     5.prepare_for_steps加载新step数据并用pbar显示进度 6.train_in_steps进行网络前向运算,计算loss,更新optimizer
|   |   |-- evaler.py                       # Evaler类:1.大部分作为实例对象的函数在eval中调用,实现了自己的数据加载,模型加载,前向运算和结果统计,数据format变换 
|   |   |                                   #     2.为Trainer类,tensorRT evaluation过程,PTQ evaluation过程 提供了eval_model函数调用,统计evaluation结果的map和速度
|   |   `-- inferer.py                      # inferer类:包括推断过程的预处理(大小变换),RepVgg旁路结构向conv变换的deploy,模型输出变换到原图,结果作图显示
|   |-- data
|   |   |-- data_augment.py                 # 1.为TrainValDataset类提供扩增变换函数 2.letterbox在train,evaluation,inferer过程中将图像大小变换到适合网络的大小
|   |   |-- data_load.py                    # 为train,evaluation过程提供dataloader,通过TrainValDataLoader加载TrainValDataset实现
|   |   |-- datasets.py                     # TrainValDataset类:实现voc和yolo数据集的图像和label存在检验,读取,扩增变换并最终转换成yolo format
|   |   |-- vis_dataset.py                  # voc数据集图像+框的显示
|   |   `-- voc2yolo.py                     # 转换voc数据集成yolo格式
|   |-- layers
|   |   |-- common.py                       # 基本模型单元结构
|   |   `-- dbb_transforms.py               
|   |-- models
|   |   |-- efficientrep.py                 # 多种结构Backbone网络
|   |   |-- effidehead.py                   # 多种结构Head网络
|   |   |-- end2end.py                      # 进行end2end export 模型时,为tensorRT和ONNX提供nms算子结构
|   |   |-- heads
|   |   |   |-- effidehead_distill_ns.py    # qat过程时,比一般的Head增加一条offset框回归的分支
|   |   |   `-- effidehead_fuseab.py        # anchor-aided training(AAT)的Head网络,对框回归包含anchor-base和anchor-free两条分支
|   |   |-- losses                          
|   |   |   |-- loss.py                     # 损失函数的实现,分类loss采用VFL,box_reg_loss采用DFL,iou_loss采用giou
|   |   |   |-- loss_distill.py             # yolo大网络,进行qat时loss的计算,几乎和loss_distill_ns一样
|   |   |   |-- loss_distill_ns.py          # yolos,yolon进行qat时loss的计算,计算学生网络的VFL,DFL,giou_loss.计算教师网络和学生网络之间分类网络输出的KL散度,feature_map中各像素的KL散度
|   |   |   `-- loss_fuseab.py              # AAT时loss的计算,和loss.py几乎一样,在engine.py的train_in_steps中anchor-base和anchor-free两条分支的loss各计算一遍
|   |   |-- reppan.py                       # 多种结构Neck(PAN)网络
|   |   `-- yolo.py                         # Model类:创建模型实例,初始化模型,加载模型权重的位置,并提供forward模型调用的前向计算.
|   |-- solver
|   |   `-- build.py                        # 为train过程提供optimizer
|   `-- utils
|       |-- Arial.ttf
|       |-- RepOptimizer.py
|       |-- checkpoint.py                   # train过程中checkpoint的加载与保存
|       |-- config.py                       # 实现adddict类型的配置文件加载
|       |-- ema.py                          # ModelEMA类:在训练过程中实现权重和参数的滑动平均更新
|       |-- envs.py                         # 运行环境读入,GPU选择
|       |-- events.py                       # 实现yaml文件加载,将evaluation的统计结果写入log,在tensorboard上模型结果显示
|       |-- figure_iou.py                   # 多种iou_type下的IOUloss计算
|       |-- general.py                      # 提供bbox框的数据表示形式变换,xyxy下的iou计算,last_checkpoint加载,对save路径返回不存在的并加以新建,模型下载
|       |-- metrics.py                      # 为evaluation过程提供结果统计和统计结果作图
|       |-- nms.py                          # box-nms计算
|       `-- torch_utils.py                  # 提供cuda任务时钟,权重初始化,conv+bn权值融合,模型参数量计算
|-- docs/                                   # The guidence explains how to use the ro repository
|-- runs/                                   # 任务运行过程中及结果的保存位置
`-- hubconf.py

```