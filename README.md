
# SoccerNet Ball Action Spotting Challenge 2024 Baseline

This repo is a fork of the [baseline for the SoccerNet Ball Action Spotting Challenge 2024](https://github.com/recokick/ball-action-spotting) for the [SoccerNet Ball Action Spotting Challenge 2024](https://www.soccer-net.org/challenges/2024).

This repo stores the solution of SoccerNet Ball Action Spotting Challenge 2024 by Team Ai4sports. We tried some methods to improve the ability of classification.

## Quick setup and start

### Requirements

* Linux (tested on Ubuntu 20.04)
* NVIDIA GPU (pipeline tuned for RTX 3080)
* NVIDIA Drivers >= 520, CUDA >= 11.8
* [Docker](https://docs.docker.com/engine/install/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Run

Clone the repo and enter the folder.

```bash
git clone git@github.com:recokick/ball-action-spotting.git
cd ball-action-spotting
```

Build a Docker image and run a container.


```bash
make
```

From now on, you should run all commands inside the docker container.

Download `sampling_weights_001` and `action_sampling_weights_002` from author's [Google Drive](https://drive.google.com/drive/folders/1mIu62cIdsRn3W4o1E5vRR8V5Q1B6HHoz?usp=sharing) and copy the files to the [data](data) directory so that the folder structure is as follows:

```
data
├── action
│   ├── experiments
│   │   └── action_sampling_weights_002
│   └── predictions
│       └── action_sampling_weights_002
├── ball_action
│   ├── experiments
│   │   └── sampling_weights_001
│   └── predictions
│       └── sampling_weights_001
├── readme_images
└── soccernet
    └── spotting-ball-2024
        └── england_efl
```

OR

Download the Ball Action Spotting 2023 dataset and Action Spotting 2023 dataset if you want to train the models from scratch.
To get the password, you must fill NDA ([link](https://www.soccer-net.org/data)).

Now you can train models and use them to predict games.
To reproduce the final solution, you can use the following commands (for the `--experiment sampling_weights_001` parts of the steps you might want to change the constant `soccernet_dir` in src/ball_action/constants.py to `soccernet_dir / "spotting-ball-2023"`):

```bash
# Train and predict basic experiment on all folds
python scripts/ball_action/train.py --experiment sampling_weights_001
python scripts/ball_action/predict.py --experiment sampling_weights_001

# Training on Action Spotting Challenge dataset
python scripts/action/train.py --experiment action_sampling_weights_002

# Transfer learning
python scripts/ball_action/train.py --experiment ball_tuning_001
python scripts/ball_action/predict.py --experiment ball_tuning_001
python scripts/ball_action/evaluate.py --experiment ball_tuning_001
python scripts/ball_action/predict.py --experiment ball_tuning_001 --challenge
python scripts/ball_action/ensemble.py --experiments ball_tuning_001 --challenge
# To train models without the use of the test subset of the data in training use argument --folds train 

# Spotting results will be there
cd data/ball_action/predictions/ball_tuning_001/challenge/ensemble/
zip results_spotting.zip ./*/*/*/results_spotting.json
```

