# [Unified Contrastive Learning in Image-Text-Label Space](https://arxiv.org/abs/2204.03610)

This is the official Pytorch implementation of UniCL:

["**Unifiled Contrastive Learning in Image-Text-Label Space. CVPR 2022**"](https://arxiv.org/abs/2204.03610) by [Jianwei Yang*](https://jwyang.github.io/), [Chunyuan Li*](https://chunyuan.li/), [Pengchuan Zhang*](https://pzzhang.github.io/pzzhang/), [Bin Xiao*](https://www.microsoft.com/en-us/research/people/bixi/), [Ce Liu](http://people.csail.mit.edu/celiu/), [Lu Yuan](https://scholar.google.com/citations?user=k9TsUVsAAAAJ&hl=en) and [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fjfgao%2F).

## Introduction

<p align="center">
  <img src="figures/unified_cv.png" width=98%/>
</p>

In this paper, we introduce a new perspective on commonly used image-label and image-text data by residing them in an image-text-label space. In this space, a new learning paradigm, called **Unified Contrastive Learning (UniCL)** with a single learning objective is proposed to seamlessly prompt the synergy of two data types. We demonstrate that UniCL is an effective way of learning **semantically rich yet discriminative representations**, universally for image recognition in zero-shot, linear-probe, fully finetuning and transfer learning scenarios. When scaled up to billions of data, UniCL can exclusively learn a powerful visual-semantic representation supporting dozens of downstream tasks shown in [Florence](https://arxiv.org/pdf/2111.11432v1.pdf).

We make the comparisons between UniCL with coventional learning methods below:

<p align="center">
  <img src="figures/unicl_comparison.png" width=98%/>
</p>

## Updates

* :collision: [10/05/2022] How do we use the pretrainied UniCL checkpoints? Beyond the zero-shot classification shown in our paper, we can use them for object detection. Now [RegionCLIP](https://github.com/microsoft/RegionCLIP/tree/zero-shot) supports to use pretrained UniCL transformer models, such as Swin, ViT for open-vocabulary object detection without any finetuning. Check it out!
* [08/19/2022] Organizing ECCV Workshop [*Computer Vision in the Wild (CVinW)*](https://computer-vision-in-the-wild.github.io/eccv-2022/), where two challenges are hosted to evaluate the zero-shot, few-shot and full-shot performance of pre-trained vision models in downstream tasks:
  - [``*Image Classification in the Wild (ICinW)*''](https://eval.ai/web/challenges/challenge-page/1832/overview) Challenge evaluates on 20 image classification tasks.
  - [``*Object Detection in the Wild (ODinW)*''](https://eval.ai/web/challenges/challenge-page/1839/overview) Challenge evaluates on 35 object detection tasks.

$\qquad$ [ <img src="https://computer-vision-in-the-wild.github.io/eccv-2022/static/eccv2022/img/ECCV-logo3.png" width=10%/> [Workshop]](https://computer-vision-in-the-wild.github.io/eccv-2022/)    $\qquad$    [<img src="https://evalai.s3.amazonaws.com/media/logos/4e939412-a9c0-46bd-9797-5ba0bd0a9095.jpg" width=10%/> [IC Challenge] ](https://eval.ai/web/challenges/challenge-page/1832/overview)
$\qquad$    [<img src="https://evalai.s3.amazonaws.com/media/logos/3a31ae6e-a990-48fb-b2c3-1e7da9d17a20.jpg" width=10%/> [OD Challenge] ](https://eval.ai/web/challenges/challenge-page/1839/overview)



* [06/19/2022] Released the evaluation benchmark used in UniCL, [ELEVATER](https://arxiv.org/abs/2204.08790), which contains 20 downstream image classification tasks. More info: [[Benchmark]](https://computer-vision-in-the-wild.github.io/ELEVATER/) [[Toolkit]](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC)  [[Paper]](https://arxiv.org/abs/2204.08790)
* [06/04/2022] Checkout out our [Huggingface Gradio demo](https://huggingface.co/spaces/CVPR/unicl-zero-shot-img-recog).
* [05/21/2022] Released pretrained model and zero-shot evaluation on ImageNet-1k.

## Benchmarking

### Image-label training augmented by image-text pairs

| Model | Training Set | Top-1 on IN-1K | ZS on 14 datasets | Download
| :----: | :---: | :---: | :---: | :---: |
| Swin-T | IN-1K | 79.9 | 30.2 | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in1k.pth)/[config](configs/unicl_swin_tiny.yaml)
| Swin-T | IN-1K + GCC-3M | 80.2 | 39.0 | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in1k_gcc3m.pth)/[config](configs/unicl_swin_tiny.yaml)
| Swin-T | IN-1K + GYFCC-14M | 81.1 | 40.0 | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in1k_yfcc14m.pth)/[config](configs/unicl_swin_tiny.yaml)
| Swin-T | IN-1K + GCC-15M | 81.8 | 45.1 | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in1k_gcc15m.pth)/[config](configs/unicl_swin_tiny.yaml)

Note that all the above models are trained **without** strong data augmentations like mixup and cutmix.

### Image-text learning augmented by image-label data

| Model | Training Set | ZS on IN-1K | ZS on 14 datasets | ZS on 20 datasets | Download
| :----: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | YFCC-14M | 30.1 | 36.3 | - | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/yfcc14m.pth)/[config](configs/unicl_swin_tiny.yaml)
| Swin-T | IN-21K | 28.5 | 37.8 | - | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in21k.pth)/[config](configs/unicl_swin_tiny.yaml)
| Swin-T | IN-22K | 66.8 | 38.9 | - | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in22k.pth)/[config](configs/unicl_swin_tiny.yaml)
| Swin-T | IN-21K (half) + YFCC-14M (half) | 36.4 | 45.5 | - | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in21k_yfcc14m_half.pth)/[config](configs/unicl_swin_tiny.yaml)
| Swin-T | IN-21K + YFCC-14M | 40.5 | 49.1 | - | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in21k_yfcc14m.pth)/[config](configs/unicl_swin_tiny.yaml)
| Swin-B | IN-21K | 29.9 | 42.4 | - | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in21k_swin_base.pth)/[config](configs/unicl_swin_base.yaml)
| Swin-B | IN-21K (half) + YFCC-14M (half) | 41.1 | 48.5 | - | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in21k_yfcc14m_half_swin_base.pth)/[config](configs/unicl_swin_base.yaml)
| Swin-B | IN-21K + YFCC-14M | 44.3 | 52.2 | - | [ckpt](https://projects4jw.blob.core.windows.net/unicl/release/in21k_yfcc14m_swin_base.pth)/[config](configs/unicl_swin_base.yaml)
| Swin-B | IN-21K + GCC-15M + YFCC-14M | 52.2 | - | 43.2 | [ckpt]()/[config](configs/unicl_swin_base.yaml)
| Focal-B | IN-21K + GCC-15M + YFCC-14M | 54.2 | - | 44.0 | [ckpt]()/[config]()

**NOTE**: Setting "ZS on 20 datasets" is used in the ICinW benchmark.

## Getting Started

### Installation

Please follow [INSTALL.md](./INSTALL.md) for installation.

### Data preparation

Please following [DATA.md](./DATA.md) for data preparation.

### Evaluation

To evaluate a pre-trained UniCL on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

For example, to evaluate the UniCL-Swin-Tiny trained on YFCC-14M with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/unicl_swin_tiny.yaml --resume yfcc14m.pth --data-path <imagenet-path>
```

###  The Image Classification in the Wild Benchmark
Interested in evaluating UniCL for downstream image classification tasks, and comparing performance on the same task suite? We release [ELEVATER](https://computer-vision-in-the-wild.github.io/ELEVATER/) benchmark, which has 20 downstream image classification tasks. The [software toolkit](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC) is also released to ease the process to onboad new models. It will be hosted as a challenge at the [CV in the Wild Workshop @ ECCV 2022](https://computer-vision-in-the-wild.github.io/eccv-2022/). We hope our benchmark and toolkit can encourage the community to solve the challenge of image classification in the wild!

Please see more instructions: [[Benchmark]](https://computer-vision-in-the-wild.github.io/ELEVATER/) [[Toolkit]](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC) [[Paper]](https://arxiv.org/abs/2204.08790)

## Citation

If you find this repo useful to your project, please consider to cite it with following bib:

    @misc{yang2022unified,
        title={Unified Contrastive Learning in Image-Text-Label Space}, 
        author={Jianwei Yang and Chunyuan Li and Pengchuan Zhang and Bin Xiao and Ce Liu and Lu Yuan and Jianfeng Gao},
        year={2022},
        eprint={2204.03610},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

## Acknowledgement

Our codebase is built based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Focal Transformer](https://github.com/microsoft/Focal-Transformer) and [FocalNet](https://github.com/microsoft/FocalNet).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
