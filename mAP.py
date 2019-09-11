import torch
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
from vision.utils import box_utils, measurements
from tqdm import tqdm
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.ssd import MatchPrior
from torch.utils.data import DataLoader
from vision.datasets.sku_dataset import SKUDataset
from vision.nn.multibox_loss import MultiboxLoss
from pathlib import Path
import numpy as np

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.sku_dataset import SKUDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import logging
import sys
from tqdm import tqdm
import cv2





def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            # all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
            all_gt_boxes[class_index][image_id] = all_gt_boxes[class_index][image_id].clone().detach()
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


def mAP(dataset, net_test, device):
    """
        Three ways to get image data:

        1. Get image with dataset.get_image(i)
        2. Read directly by cv2.imread(image_path)
        In essence, 1. and 2. are the same.
        Here, the cv2.cvtColor(image, cv2.COLOR_BGR2RGB) is involved, which will cause the dataloader to enter the infinite wait under the condition that the linux system is set to num_workers>0, and the thread is stuck.

        3. Use the dataloader of batch_size=1 to read the image data
        Although the problems solved by 1 and 2 are solved, but the time is nearly three times that of 1 and 2.

        4. Use the least time-consuming method 2, while removing the cv2.cvtColor(image, cv2.COLOR_BGR2RGB) (which will cause the ap value to be higher than before) to achieve image data reading.
        TODO : Why cv2.cvtColor(image, cv2.COLOR_BGR2RGB) will cause multithreading to die ? How to solve ?
        """
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    predictor = create_mobilenetv3_ssd_lite_predictor(net_test, candidate_size=200, device=device)
    class_names = dataset.class_names
    # print(f'mAP-class_names : {len(class_names)}')
    results = []
    # # 1. Get image with dataset.get_image(i)
    # for i in range(len(dataset)):
    #     image = dataset.get_image(i)
    # # 2. Read directly by cv2.imread(image_path)
    # root = dataset.root
    # ids = dataset.ids
    # for i in range(len(dataset)):
    #     id = ids[i]
    #     image_file = root / f"JPEGImages/{id}.jpg"
    #     image = cv2.imread(str(image_file))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # 3. Use the dataloader of batch_size=1 to read the image data
    loader = DataLoader(dataset, 1, num_workers=0)
    for i, image in tqdm(enumerate(loader)):
        image = image[0].numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # 4. Use the least time-consuming method 2, while removing the cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # root = dataset.root
    # ids = dataset.ids
    # for i in range(len(dataset)):
    #     id = ids[i]
    #     image_file = root / f"JPEGImages/{id}.jpg"
    #     image = cv2.imread(str(image_file))
    #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels, probs = predictor.predict(image)
        if labels.size(0) == 0:
            continue
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat((
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ), dim=1))

    results = torch.cat(results)

    aps = []
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        id_list = []
        score_list = []
        box_list = []
        sub = results[results[:, 1] == class_index, :]
        for i in range(sub.size(0)):
            prob_box = sub[i, 2:].numpy()
            image_id = dataset.ids[int(sub[i, 0])]
            id_list.append(image_id)
            score_list.append(prob_box[0])
            box_list.append(prob_box[1:])

        ap = measurements.compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            id_list,
            score_list,
            box_list,
            iou_threshold=0.5,
            use_2007_metric=True
        )
        aps.append(ap)
        # print(f"{class_name}: {ap}")
    # print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    return aps


# if __name__ == '__main__':
#     from vision.ssd.mobilenet_v3_ssd_lite_onnx import create_mobilenetv3_ssd_lite
#     sku_model_path = 'models/ssd_mb3l_348.pth'
#     cf_model_path = 'models/large.t7'
#     cfl_path = 'cifar-large.pth'
#     cf_model = torch.load(cf_model_path)['model']
#     sku_model = torch.load(sku_model_path)
#     cfl_model = torch.load(cfl_path)
#     cf_list = list(cf_model)
#     cfl_list = list(cfl_model)
#     sku_list = list(sku_model)
#     print(cfl_list==sku_list)
#     # print(len(cf_model))
#     # print(len(sku_model))
#     # print(cf_list[345:])
#     # print(sku_list[345:])
#     # ssd_model = {}
#     # for i, layer in enumerate(cf_model):
#     #     try:
#     #         print(f'{layer} - {sku_list[i]}')
#     #
#     #         ssd_model[sku_list[i]] = cf_model[layer]
#     #     except IndexError as e:
#     #         print(i)
#     #         print(e)
#     #         print(len(ssd_model))
#     #         torch.save(ssd_model, 'cifar-large.pth')
#     #
#     #         print(len(torch.load('cifar-large.pth')))
#     #         break
#
#     # for d in ds:
#     #     print(d)
#     # net = create_mobilenetv1_ssd(31)
#     # print(net)