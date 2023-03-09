import json
import argparse
from pathlib import Path
from datetime import timedelta

from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import torch
import argus

from src.ball_action.indexes import StackIndexesGenerator
from src.frame_fetchers import NvDecFrameFetcher
from src.utils import get_best_model_path, get_video_info, normalize_tensor_frames
from src.ball_action import constants


NUM_HALVES = 2
RESOLUTION = "720p"
INDEX_SAVE_ZONE = 1
POSTPROCESS_PARAMS = {
    "gauss_sigma": 3.0,
    "height": 0.3,
    "distance": 15,
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    return parser.parse_args()


def read_until_last(frame_fetcher: NvDecFrameFetcher,
                    frame_index2frame: dict,
                    frame_buffer_size: int,
                    last_frame_index: int):
    while True:
        frame = frame_fetcher.fetch_frame()
        frame_index = frame_fetcher.current_index
        frame_index2frame[frame_index] = frame
        del_frame_index = frame_index - frame_buffer_size
        if del_frame_index in frame_index2frame:
            del frame_index2frame[del_frame_index]
        if frame_index == last_frame_index:
            break


def post_processing(frame_indexes: list[int],
                    predictions: np.ndarray,
                    gauss_sigma: float,
                    height: float,
                    distance: int) -> tuple[list[int], list[float]]:
    predictions = gaussian_filter(predictions, gauss_sigma)
    peaks, _ = find_peaks(predictions, height=height, distance=distance)
    confidences = predictions[peaks].tolist()
    action_frame_indexes = (peaks + frame_indexes[0]).tolist()
    return action_frame_indexes, confidences


def get_raw_predictions(model: argus.Model, video_path: Path) -> tuple[list[int], np.ndarray, dict]:
    video_info = get_video_info(video_path)
    frame_count = video_info["frame_count"]
    frame_index2frame: dict[int, torch.Tensor] = dict()
    frame_stack_size = model.params["frame_stack_size"]
    frame_stack_step = model.params["frame_stack_step"]
    frame_buffer_size = frame_stack_size * frame_stack_step
    indexes_generator = StackIndexesGenerator(frame_stack_size, frame_stack_step)
    frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=model.device.index)
    frame_fetcher.num_frames = frame_count

    min_frame_index = indexes_generator.clip_index(0, frame_count, INDEX_SAVE_ZONE)
    max_frame_index = indexes_generator.clip_index(frame_count, frame_count, INDEX_SAVE_ZONE)
    frame_index2prediction = dict()
    for frame_index in tqdm(range(min_frame_index, max_frame_index + 1)):
        frame_indexes = indexes_generator.make_stack_indexes(frame_index)
        read_until_last(frame_fetcher, frame_index2frame, frame_buffer_size, max(frame_indexes))
        frames = torch.stack([frame_index2frame[i] for i in frame_indexes], dim=0)
        frames = normalize_tensor_frames(frames.unsqueeze(0))
        prediction = model.predict(frames)[0].cpu().numpy()
        frame_index2prediction[frame_index] = prediction

    frame_indexes = sorted(frame_index2prediction.keys())
    raw_predictions = np.stack([frame_index2prediction[i] for i in frame_indexes], axis=0)
    return frame_indexes, raw_predictions, video_info


def predict_video(model: argus.Model,
                  half: int,
                  game_dir: Path,
                  game_prediction_dir: Path) -> tuple[dict[str, tuple], dict]:
    video_path = game_dir / f"{half}_{RESOLUTION}.mkv"
    print(f"Predict video:", video_path)
    frame_indexes, raw_predictions, video_info = get_raw_predictions(model, video_path)
    raw_predictions_path = game_prediction_dir / f"{half}_raw_predictions.npz"
    np.savez(
        raw_predictions_path,
        frame_indexes=frame_indexes,
        raw_predictions=raw_predictions,
    )
    print("Raw predictions saved to", raw_predictions_path)

    class2actions = dict()
    for cls, cls_index in constants.class2target.items():
        class2actions[cls] = post_processing(
            frame_indexes, raw_predictions[:, cls_index], **POSTPROCESS_PARAMS
        )
        print(f"Predicted {len(class2actions[cls][0])} {cls} actions")

    return class2actions, video_info


def predict_game(model: argus.Model, game: str, prediction_dir: Path):
    game_dir = constants.ball_action_soccernet_dir / game
    game_prediction_dir = prediction_dir / game
    game_prediction_dir.mkdir(parents=True, exist_ok=True)
    print(f"Predict game:", game)

    half2class_actions = dict()
    halves = list(range(1, NUM_HALVES + 1))
    halv2video_info = dict()
    for half in halves:
        class_actions, video_info = predict_video(model, half, game_dir, game_prediction_dir)
        half2class_actions[half] = class_actions
        halv2video_info[half] = video_info

    results_spotting = {
        "UrlLocal": game,
        "predictions": list(),
    }

    for half in halves:
        video_info = halv2video_info[half]
        for cls, (frame_indexes, confidences) in half2class_actions[half].items():
            for frame_index, confidence in zip(frame_indexes, confidences):
                position = round(frame_index / video_info["fps"] * 1000)
                _, minutes, seconds = str(timedelta(milliseconds=position)).split(":")
                prediction = {
                    "gameTime": f"{half} - {minutes}:{seconds}",
                    "label": cls,
                    "position": str(position),
                    "half": str(half),
                    "confidence": str(confidence),
                }
                results_spotting["predictions"].append(prediction)
    results_spotting["predictions"] = sorted(
        results_spotting["predictions"],
        key=lambda pred: int(pred["position"])
    )

    results_spotting_path = game_prediction_dir / "results_spotting.json"
    with open(results_spotting_path, "w") as outfile:
        json.dump(results_spotting, outfile, indent=4)
    print("Spotting results saved to", results_spotting_path)


def predict_games(experiment: str, split: str, gpu_id: int):
    print(f"Predict games: {experiment=}, {split=}, {gpu_id=}")
    experiment_dir = constants.experiments_dir / experiment
    model_path = get_best_model_path(experiment_dir)
    print("Model path:", model_path)
    model = argus.load_model(model_path, device=f"cuda:{gpu_id}")
    prediction_dir = constants.predictions_dir / experiment / split
    if not prediction_dir.exists():
        prediction_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {prediction_dir} already exists.")
    games = constants.split2games[split]
    for game in games:
        predict_game(model, game, prediction_dir)


if __name__ == "__main__":
    args = parse_arguments()
    predict_games(args.experiment, args.split, args.gpu_id)