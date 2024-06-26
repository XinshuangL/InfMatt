# InfMatt: Image Matting with Infinite Refinement Iterations

Official implementation of the ICANN 2024 paper InfMatt: Image Matting with Infinite Refinement Iterations. 

<p align="middle">
    <img src="illustration.png">
</p>

## Installation
Plese download the pre-trained weights from [MDEQ](https://github.com/locuslab/mdeq) and [InSPyReNet](https://github.com/plemeri/InSPyReNet) and put them in `./pretrained/`.

## Evaluation
### Data Preparation
Please organize the datasets as follows.

    ../                         # parent directory
    ├── ./                      # current (project) directory
    ├── AMD/                    # the dataset
    │   ├── train/
    │   │   ├── fg/
    │   │   └── alpha/
    │   └── test/           
    │       ├── merged/
    │       └── alpha_copy/
    ├── ...

### Train & Test
    
```sh
python main.py --dataset [dataset_name]
```

## Acknowledgement
Thanks to the code base from [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting), [MODNet](https://github.com/ZHKKKe/MODNet), [MGMatting](https://github.com/yucornetto/MGMatting), [MatteFormer](https://github.com/webtoon/matteformer), [MDEQ](https://github.com/locuslab/mdeq), [InSPyReNet](https://github.com/plemeri/InSPyReNet)
