from src.ball_action import constants
from src.utils import get_lr


image_size = (1280, 736)
batch_size = 4
base_lr = 3e-4
frame_stack_size = 15

config = dict(
    image_size=image_size,
    batch_size=batch_size,
    base_lr=base_lr,
    min_base_lr=base_lr * 0.01,
    ema_decay=0.999,
    max_targets_window_size=15,
    train_epoch_size=6000,
    train_sampling_weights=dict(
        action_window_size=9,
        action_prob=0.5,
        # TODO 0101 增加14*weigths参数，对应加参数的位置我就不一个一个标注了，自己找
        action_weights={
            "PASS" : 0.196,
            "DRIVE" : 0.21,
            "HEADER" : 0.518,
            "HIGH PASS" : 0.49000000000000005,
            "OUT" : 0.5880000000000001,
            "CROSS" : 0.854,
            "THROW IN" : 0.728,
            "SHOT" : 1.064,
            "BALL PLAYER BLOCK" : 0.924,
            "PLAYER SUCCESSFUL TACKLE" : 1.596,
            "FREE KICK" : 3.01,
            "GOAL" : 3.822,
        },
        # TODO 0101
        pred_experiment="",
        clear_pred_window_size=9,
    ),
    metric_accuracy_threshold=0.5,
    num_nvdec_workers=3,
    num_opencv_workers=1,
    num_epochs=[6, 30],
    stages=["warmup", "train"],
    argus_params={
        "nn_module": ("multidim_stacker", {
            "model_name": "tf_efficientnetv2_b0.in1k",
            "num_classes": constants.num_classes,
            "num_frames": frame_stack_size,
            "stack_size": 3,
            "index_2d_features": 4,
            "pretrained": True,
            "num_3d_blocks": 4,
            "num_3d_features": 192,
            "expansion_3d_ratio": 3,
            "se_reduce_3d_ratio": 24,
            "num_3d_stack_proj": 256,
            "drop_rate": 0.2,
            "drop_path_rate": 0.2,
            "act_layer": "silu",
        }),
        "loss": ("focal_loss", {
            "alpha": -1.0,
            "gamma": 1.2,
            "reduction": "mean",
        }),
        "optimizer": ("AdamW", {
            "lr": get_lr(base_lr, batch_size),
        }),
        "device": ["cuda:0"],
        "image_size": image_size,
        "frame_stack_size": frame_stack_size,
        "frame_stack_step": 2,
        "amp": True,
        "iter_size": 1,
        "frames_processor": ("pad_normalize", {
            "size": image_size,
            "pad_mode": "constant",
            "fill_value": 0,
        }),
        "freeze_conv2d_encoder": False,
    },
    frame_index_shaker={
        "shifts": [-1, 0, 1],
        "weights": [0.2, 0.6, 0.2],
        "prob": 0.25,
    },
    pretrain_action_experiment="",
    pretrain_ball_experiment="",
    torch_compile={
        "backend": "inductor",
        "mode": "default",
    },
)
