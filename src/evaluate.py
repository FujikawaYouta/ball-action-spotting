import os
import json
import glob
import zipfile
import pdb

from tqdm import tqdm
import numpy as np

# from SoccerNet.Evaluation.ActionSpotting import label2vector, predictions2vector
from SoccerNet.Evaluation.ActionSpotting import average_mAP
from SoccerNet.Evaluation.utils import LoadJsonFromZip, EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, EVENT_DICTIONARY_BALL


def label2vector(labels, num_classes=17, framerate=2, version=2, EVENT_DICTIONARY={}):
    vector_size = 120 * 60 * framerate

    label_half1 = np.zeros((vector_size, num_classes))
    label_half2 = np.zeros((vector_size, num_classes))

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        # annotation at millisecond precision
        if "position" in annotation:
            frame = int(framerate * (int(annotation["position"]) / 1000))
            # annotation at second precision
        else:
            frame = framerate * (seconds + 60 * minutes)

        if version == 2:
            if event not in EVENT_DICTIONARY:
                continue
            label = EVENT_DICTIONARY[event]
        elif version == 1:
            # print(event)
            # label = EVENT_DICTIONARY_V1[event]
            if "card" in event:
                label = 0
            elif "subs" in event:
                label = 1
            elif "soccer" in event:
                label = 2
            else:
                # print(event)
                continue
        # print(event, label, half)

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size - 1)
            label_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size - 1)
            label_half2[frame][label] = value

    return label_half1, label_half2


def predictions2vector(predictions, num_classes=17, version=2, framerate=2, EVENT_DICTIONARY={}):
    vector_size = 120 * 60 * framerate

    prediction_half1 = np.zeros((vector_size, num_classes)) - 1
    prediction_half2 = np.zeros((vector_size, num_classes)) - 1

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        half = int(annotation["half"])

        frame = int(framerate * (time / 1000))

        if version == 2:
            if event not in EVENT_DICTIONARY:
                continue
            label = EVENT_DICTIONARY[event]
        elif version == 1:
            label = EVENT_DICTIONARY[event]
            # print(label)
            # EVENT_DICTIONARY_V1[l]
            # if "card" in event: label=0
            # elif "subs" in event: label=1
            # elif "soccer" in event: label=2
            # else: continue

        value = annotation["confidence"]

        if half == 1:
            frame = min(frame, vector_size - 1)
            prediction_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size - 1)
            prediction_half2[frame][label] = value

    return prediction_half1, prediction_half2

# def average_mAP(targets, detections, closests, framerate=2, deltas=np.arange(5) * 1 + 1):
#     mAP, mAP_per_class, mAP_visible, mAP_per_class_visible, mAP_unshown, mAP_per_class_unshown = delta_curve(targets,
#                                                                                                              closests,
#                                                                                                              detections,
#                                                                                                              framerate,
#                                                                                                              deltas)
#
#     if len(mAP) == 1:
#         return mAP[0], mAP_per_class[0], mAP_visible[0], mAP_per_class_visible[0], mAP_unshown[0], \
#         mAP_per_class_unshown[0]
#
#     # Compute the average mAP
#     integral = 0.0
#     for i in np.arange(len(mAP) - 1):
#         integral += (mAP[i] + mAP[i + 1]) / 2
#     a_mAP = integral / ((len(mAP) - 1))
#
#     integral_visible = 0.0
#     for i in np.arange(len(mAP_visible) - 1):
#         integral_visible += (mAP_visible[i] + mAP_visible[i + 1]) / 2
#     a_mAP_visible = integral_visible / ((len(mAP_visible) - 1))
#
#     integral_unshown = 0.0
#     for i in np.arange(len(mAP_unshown) - 1):
#         integral_unshown += (mAP_unshown[i] + mAP_unshown[i + 1]) / 2
#     a_mAP_unshown = integral_unshown / ((len(mAP_unshown) - 1))
#     a_mAP_unshown = a_mAP_unshown * 17 / 13
#
#     a_mAP_per_class = list()
#     for c in np.arange(len(mAP_per_class[0])):
#         integral_per_class = 0.0
#         for i in np.arange(len(mAP_per_class) - 1):
#             integral_per_class += (mAP_per_class[i][c] + mAP_per_class[i + 1][c]) / 2
#         a_mAP_per_class.append(integral_per_class / ((len(mAP_per_class) - 1)))
#
#     a_mAP_per_class_visible = list()
#     for c in np.arange(len(mAP_per_class_visible[0])):
#         integral_per_class_visible = 0.0
#         for i in np.arange(len(mAP_per_class_visible) - 1):
#             integral_per_class_visible += (mAP_per_class_visible[i][c] + mAP_per_class_visible[i + 1][c]) / 2
#         a_mAP_per_class_visible.append(integral_per_class_visible / ((len(mAP_per_class_visible) - 1)))
#
#     a_mAP_per_class_unshown = list()
#     for c in np.arange(len(mAP_per_class_unshown[0])):
#         integral_per_class_unshown = 0.0
#         for i in np.arange(len(mAP_per_class_unshown) - 1):
#             integral_per_class_unshown += (mAP_per_class_unshown[i][c] + mAP_per_class_unshown[i + 1][c]) / 2
#         a_mAP_per_class_unshown.append(integral_per_class_unshown / ((len(mAP_per_class_unshown) - 1)))
#
#     return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown


def evaluate(SoccerNet_path, Predictions_path, list_games, prediction_file="results_spotting.json", version=2,
             framerate=2, metric="loose", label_files="Labels-v2.json", num_classes=17, dataset="SoccerNet"):
    # evaluate the prediction with respect to some ground truth
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - list_games: games to evaluate
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()
    if dataset == "SoccerNet" and version == 1:
        EVENT_DICTIONARY = EVENT_DICTIONARY_V1
    elif dataset == "SoccerNet" and version == 2:
        EVENT_DICTIONARY = EVENT_DICTIONARY_V2
    elif dataset == "Headers":
        EVENT_DICTIONARY = {"Header": 0}
    elif dataset == "Headers-headimpacttype":
        EVENT_DICTIONARY = {"1. Purposeful header": 0, "2. Header Duel": 1,
                            "3. Attempted header": 2, "4. Unintentional header": 3, "5. Other head impacts": 4}
    elif dataset == "Ball":
        EVENT_DICTIONARY = EVENT_DICTIONARY_BALL
    elif dataset == "NewBallAction":
        EVENT_DICTIONARY = {'PASS' : 0, 'DRIVE' : 1, 'HEADER' : 2, 'HIGH PASS' : 3,
                            'OUT' : 4, 'CROSS' : 5, 'THROW IN' : 6, 'SHOT' : 7,
                            'BALL PLAYER BLOCK': 8, 'PLAYER SUCCESSFUL TACKLE' : 9,
                            'FREE KICK' : 10, 'GOAL' : 11}


    for game in tqdm(list_games):

        # # Load labels
        # if version==2:
        #     label_files = "Labels-v2.json"
        #     num_classes = 17
        # elif version==1:
        #     label_files = "Labels.json"
        #     num_classes = 3
        # if dataset == "Headers":
        #     label_files = "Labels-Header.json"
        #     num_classes = 3

        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, label_files))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        # convert labels to vector
        label_half_1, label_half_2 = label2vector(
            labels, num_classes=num_classes, version=version, EVENT_DICTIONARY=EVENT_DICTIONARY, framerate=framerate)
        # print(version)
        # print(label_half_1)
        # print(label_half_2)

        # infer name of the prediction_file
        if prediction_file is None:
            if zipfile.is_zipfile(Predictions_path):
                with zipfile.ZipFile(Predictions_path, "r") as z:
                    for filename in z.namelist():
                        #       print(filename)
                        if filename.endswith(".json"):
                            prediction_file = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(os.path.join(Predictions_path, "*/*/*/*.json")):
                    prediction_file = os.path.basename(filename)
                    # print(prediction_file)
                    break

        # Load predictions
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        # convert predictions to vector
        predictions_half_1, predictions_half_2 = predictions2vector(
            predictions, num_classes=num_classes, version=version, EVENT_DICTIONARY=EVENT_DICTIONARY,
            framerate=framerate)

        targets_numpy.append(label_half_1)
        detections_numpy.append(predictions_half_1)

        if dataset != "NewBallAction":
            targets_numpy.append(label_half_2)
            detections_numpy.append(predictions_half_2)

        closest_numpy = np.zeros(label_half_1.shape) - 1
        # Get the closest action index
        for c in np.arange(label_half_1.shape[-1]):
            indexes = np.where(label_half_1[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = label_half_1[indexes[i], c]
        closests_numpy.append(closest_numpy)

        if dataset != "NewBallAction":
            closest_numpy = np.zeros(label_half_2.shape) - 1
            for c in np.arange(label_half_2.shape[-1]):
                indexes = np.where(label_half_2[:, c] != 0)[0].tolist()
                if len(indexes) == 0:
                    continue
                indexes.insert(0, -indexes[0])
                indexes.append(2 * closest_numpy.shape[0])
                for i in np.arange(len(indexes) - 2) + 1:
                    start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                    stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                    closest_numpy[start:stop, c] = label_half_2[indexes[i], c]
            closests_numpy.append(closest_numpy)

    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])  # np.arange(1)*1 + 1
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])
        # Compute the performances

    # pdb.set_trace()

    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version == 2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version == 2 else None,
        "a_mAP_unshown": a_mAP_unshown if version == 2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version == 2 else None,
    }
    return results
