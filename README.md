## Face Super-resolution Using Stochastic Differential Equations 



<p align="center">
  <p>Input - LR</p>   <img width="150" src="https://raw.githubusercontent.com/marcelowds/sr-sde/main/lr_image.png">
  <p>Output - SR</p>   <img width="150" src="https://raw.githubusercontent.com/marcelowds/sr-sde/main/sr_generation.gif">
</p>

This project was built using a fork of [Score-SDE](https://github.com/yang-song/score_sde)


## Prepare conda environment 

```conda create -n srsde python=3.8.2```

Install requirements

```pip3 install -r requirements.txt```

Also install jax+cuda

```pip install --upgrade jax==0.2.8 jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html```

Activate conda environment

```conda activate srsde```

## Train the models

```python3 main.py --config 'configs/ve/sr_ve.py' --mode 'train' --workdir VESDE```

## SR images 

To generate Super-Resolution images from CelebaHQ without training, download the pre-trained ```SDE-VE``` model [here](https://drive.google.com/file/d/1rhdzIyWjF94QWz2jWjfwUJPOhCADPagV/view?usp=share_link), copy to ```./VESDE/checkpoints``` and run

```python3 main.py --config 'configs/ve/sr_ve.py' --mode 'sr' --workdir VESDE```

## Tfrecords

The algorithm receives images in tfrecords format. In the tfrecords folder there is a sample of 32 images from the CelebAHQ dataset.

Adjust settings and path in files ```config/default_ve_configs.py``` and ```configs/ve/sr_ve.py```.


## Citation
* M. dos Santos, R. Laroca, R. O. Ribeiro, J. Neves, H. Proença, D. Menotti, “Face Super-Resolution Using Stochastic Differential Equations”, in *Conference on Graphics, Patterns and Images (SIBGRAPI)*, pp. 216-221, Oct. 2022. [[IEEE Xplore]](https://doi.org/10.1109/SIBGRAPI55357.2022.9991799) [[arXiv]](https://arxiv.org/abs/2209.12064)

```
@inproceedings{santos2022face,
  title = {Face Super-Resolution Using Stochastic Differential Equations},
  author = {M. {dos Santos} and R. {Laroca} and R. O. {Ribeiro} and J. {Neves} and H. {Proen\c{c}a} and D. {Menotti}},
  year = {2022},
  month = {Oct},
  booktitle = {Conference on Graphics, Patterns and Images (SIBGRAPI)},
  volume = {},
  number = {},
  pages = {216-221},
  doi = {10.1109/SIBGRAPI55357.2022.9991799},
  issn = {1530-1834},
}
```
