# Summary of YOLOv6

This is a summary of yolov8 learning process. It contains some tasks implementation process, model structrue, code annotation.

The original repository link is: https://github.com/meituan/YOLOv6

Implementation of paper:
- [YOLOv6 v3.0: A Full-Scale Reloading](https://arxiv.org/abs/2301.05586)
- [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976)

## YOLOv6文件结构以及文件功能介绍

```
.
├── assets
│   ├── yolov6s_v2_reopt.pt
│   └── yolov6s_v2_scale.pt
├── configs
│   ├── base
│   │   ├── README_cn.md
│   │   ├── README.md
│   │   ├── yolov6l_base_finetune.py
│   │   ├── yolov6l_base.py
│   │   ├── yolov6m_base_finetune.py
│   │   ├── yolov6m_base.py
│   │   ├── yolov6n_base_finetune.py
│   │   ├── yolov6n_base.py
│   │   ├── yolov6s_base_finetune.py
│   │   └── yolov6s_base.py
│   ├── experiment
│   │   ├── eval_640_repro.py
│   │   ├── yolov6n_with_eval_params.py
│   │   ├── yolov6s_csp_scaled.py
│   │   ├── yolov6t_csp_scaled.py
│   │   ├── yolov6t_finetune.py
│   │   └── yolov6t.py
│   ├── qarepvgg
│   │   ├── README.md
│   │   ├── yolov6m_qa.py
│   │   ├── yolov6n_qa.py
│   │   └── yolov6s_qa.py
│   ├── repopt
│   │   ├── yolov6n_hs.py
│   │   ├── yolov6n_opt.py
│   │   ├── yolov6n_opt_qat.py
│   │   ├── yolov6s_hs.py
│   │   ├── yolov6s_opt.py
│   │   ├── yolov6s_opt_qat.py
│   │   ├── yolov6_tiny_hs.py
│   │   ├── yolov6_tiny_opt.py
│   │   └── yolov6_tiny_opt_qat.py
│   ├── yolov6l6_finetune.py
│   ├── yolov6l6.py
│   ├── yolov6l_finetune.py
│   ├── yolov6l.py
│   ├── yolov6m6_finetune.py
│   ├── yolov6m6.py
│   ├── yolov6m_finetune.py
│   ├── yolov6m.py
│   ├── yolov6n6_finetune.py
│   ├── yolov6n6.py
│   ├── yolov6n_finetune.py
│   ├── yolov6n.py
│   ├── yolov6s6_finetune.py
│   ├── yolov6s6.py
│   ├── yolov6s_finetune.py
│   └── yolov6s.py
├── data
│   ├── coco.yaml
│   ├── dataset.yaml
│   ├── images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── image3.jpg
│   ├── mydata.yaml
│   └── voc.yaml
├── deploy
│   ├── ONNX
│   │   ├── eval_trt.py
│   │   ├── export_onnx.py
│   │   ├── OpenCV
│   │   │   ├── coco.names
│   │   │   ├── README.md
│   │   │   ├── sample.jpg
│   │   │   ├── yolo.py
│   │   │   ├── yolov5
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   └── yolov5.cpp
│   │   │   ├── yolov6
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   └── yolov6.cpp
│   │   │   ├── yolo_video.py
│   │   │   ├── yolox
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   └── yolox.cpp
│   │   │   └── yolox.py
│   │   ├── README.md
│   │   ├── YOLOv6-Dynamic-Batch-onnxruntime.ipynb
│   │   └── YOLOv6-Dynamic-Batch-tensorrt.ipynb
│   ├── OpenVINO
│   │   ├── export_openvino.py
│   │   └── README.md
│   └── TensorRT
│       ├── calibrator.py
│       ├── CMakeLists.txt
│       ├── eval_yolo_trt.py
│       ├── logging.h
│       ├── onnx_to_trt.py
│       ├── Processor.py
│       ├── README.md
│       ├── visualize.py
│       └── yolov6.cpp
├── docs
│   ├── Test_speed.md
│   ├── Train_coco_data.md
│   ├── Train_custom_data.md
│   ├── Tutorial of Quantization.md
│   ├── tutorial_repopt.md
│   └── tutorial_voc.ipynb
├── hubconf.py
├── inference.ipynb
├── LICENSE
├── README_cn.md
├── README.md
├── requirements.txt
├── runs
├── tools
│   ├── eval.py
│   ├── infer.py
│   ├── partial_quantization
│   │   ├── eval.py
│   │   ├── eval.yaml
│   │   ├── partial_quant.py
│   │   ├── ptq.py
│   │   ├── README.md
│   │   ├── sensitivity_analyse.py
│   │   └── utils.py
│   ├── qat
│   │   ├── onnx_utils.py
│   │   ├── qat_export.py
│   │   ├── qat_utils.py
│   │   └── README.md
│   ├── quantization
│   │   ├── mnn
│   │   │   └── README.md
│   │   ├── ppq
│   │   │   ├── ProgramEntrance.py
│   │   │   └── write_qparams_onnx2trt.py
│   │   └── tensorrt
│   │       ├── post_training
│   │       │   ├── Calibrator.py
│   │       │   ├── LICENSE
│   │       │   ├── onnx_to_tensorrt.py
│   │       │   ├── quant.sh
│   │       │   └── README.md
│   │       ├── requirements.txt
│   │       └── training_aware
│   │           └── QAT_quantizer.py
│   └── train.py
├── turtorial.ipynb
├── weights
│   └── yolov6n.pt
├── yolov6
│   ├── assigners
│   │   ├── anchor_generator.py
│   │   ├── assigner_utils.py
│   │   ├── atss_assigner.py
│   │   ├── __init__.py
│   │   ├── iou2d_calculator.py
│   │   └── tal_assigner.py
│   ├── core
│   │   ├── engine.py
│   │   ├── evaler.py
│   │   ├── inferer.py
│   ├── data
│   │   ├── data_augment.py
│   │   ├── data_load.py
│   │   ├── datasets.py
│   │   ├── vis_dataset.py
│   │   └── voc2yolo.py
│   ├── __init__.py
│   ├── layers
│   │   ├── common.py
│   │   ├── dbb_transforms.py
│   ├── models
│   │   ├── efficientrep.py
│   │   ├── effidehead.py
│   │   ├── end2end.py
│   │   ├── heads
│   │   │   ├── effidehead_distill_ns.py
│   │   │   ├── effidehead_fuseab.py
│   │   ├── losses
│   │   │   ├── loss_distill_ns.py
│   │   │   ├── loss_distill.py
│   │   │   ├── loss_fuseab.py
│   │   │   ├── loss.py
│   │   ├── reppan.py
│   │   └── yolo.py
│   ├── solver
│   │   ├── build.py
│   └── utils
│       ├── Arial.ttf
│       ├── checkpoint.py
│       ├── config.py
│       ├── ema.py
│       ├── envs.py
│       ├── events.py
│       ├── figure_iou.py
│       ├── general.py
│       ├── metrics.py
│       ├── nms.py
│       ├── RepOptimizer.py
│       └── torch_utils.py
├── yolov6n.onnx
└── yolov6n.pt

51 directories, 214 files
```