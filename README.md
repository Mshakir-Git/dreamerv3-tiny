# tiny-dreamerv3

What is Working
1: World Model/Actor/Crtic forward/Backward pass (Slow)(Can be buggy)
2: Video Prediction (Tensorboard)

Needs to be Added
1: TinyJit


## Instructions

### Method 1: Manual

Get dependencies with python 3.9:
```
pip install -r requirements.txt
```
(Try diifferent version of gym if gym fails)


## ROMs

In order to import ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the `.rar` file.  Once you've done that, run:

`python -m atari_py.import_roms <path to folder>`

This should print out the names of ROMs as it imports them.  The ROMs will be copied to your `atari_py` installation directory.

## Pong

```
python dreamer.py --configs atari100k --task atari_pong --logdir ./logdir/pong  
```

## Check Results

```
tensorboard --logdir ./logdir/pong 
```


# OLD Readme

## dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

### Method 1: Manual

Get dependencies with python 3.9:
```
pip install -r requirements.txt
```
Run training on DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```
### Method 2: Docker

Please refer to the Dockerfile for the instructions, as they are included within.

## Benchmarks
So far, the following benchmarks can be used for testing.
| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Minecraft](https://github.com/minerllabs/minerl) | Image and State |Discrete |100M| Vast 3D open world.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

## Results
#### DMC Proprio
![dmcproprio](https://github.com/NM512/dreamerv3-torch/assets/70328564/0d3f4d44-d487-4097-bed1-3f0573aaf902)
#### DMC Vision
![dmcvision](https://github.com/NM512/dreamerv3-torch/assets/70328564/162c24d0-5f06-4ef6-bbc3-202221f6ad11)
#### Atari 100k
![atari100k](https://github.com/NM512/dreamerv3-torch/assets/70328564/0da6d899-d91d-44b4-a8c4-d5b37413aa11)

#### Crafter
<img src="https://github.com/NM512/dreamerv3-torch/assets/70328564/a0626038-53f6-4300-a622-7ac257f4c290" width="300" height="150" />

## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
