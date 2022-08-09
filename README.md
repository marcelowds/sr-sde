## Paper Title

<p align="center">
  <img width="250" src="https://raw.githubusercontent.com/marcelowds/sr-sde/main/sr_generation.gif">
</p>


Prepare conda environment 

```conda create -n srsde python=3.8.2```

Install requirements

```pip3 install -r requirements.txt```

Also install jax+cuda

```pip install --upgrade jax==0.2.8 jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html```

Activate conda environment

```conda activate srsde```

Train the models

```python3 main.py --config 'configs/ve/sr_ve.py' --mode 'train' --workdir VESDE```

To generate Super-Resolution images without training, download the pre-trained model in ```url```, copy to ```./VESDE/checkpoits``` and run

```python3 main.py --config 'configs/ve/sr_ve.py' --mode 'sr' --workdir VESDE```
