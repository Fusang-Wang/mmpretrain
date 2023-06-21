import argparse
import csv
import os
from collections import namedtuple
import torch
import numpy as np
import json
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ImageNet subset lists provided by SimCLR into '
        'the required format in MMPretrain.')
    parser.add_argument(
        'data_root', help='Input list file, downloaded from SimCLR github repo.')
    parser.add_argument(
        'output', help='Output list file with the required format.')
    args = parser.parse_args()
    return args


def load_csv(data_root: str, filename: str, header=None):
    CSV = namedtuple("CSV", ["header", "index", "data"])
    with open(os.path.join(data_root, filename)) as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

    if header:
        headers = data[header]
        data = data[header + 1:]
    else:
        headers = []

    indices = [row[0] for row in data]
    data = [row[1:] for row in data]
    data_int = [list(map(int, i)) for i in data]

    return CSV(headers, indices, torch.tensor(data_int))


def convert2dict(splits, attr, dataroot: str, output: str, wanted_labels: list, split: str):
    # splits = load_csv(data_root, "Anno/list_eval_partition.txt")
    # identity = load_csv(data_root, "Anno/identity_CelebA.txt")
    # attr = load_csv(data_root, "Anno/list_attr_celeba.txt", header=1)
    split_map = {"train": 0, "valid": 1, "test": 2}
    split_num = split_map[split]
    split_mask = torch.flatten(splits.data == split_num)

    img_list = [attr.index[i] for i in range(len(attr.index)) if split_mask[i]]
    attrs = attr.data[split_mask, ...]

    assert len(img_list) == attrs.size(0)
    wanted_attrs = attrs[..., wanted_labels]
    res = {}
    meta_info = {}
    names = np.array(attr.header)[wanted_labels]
    meta_info["classes"] = list(names)
    data_list = []
    for img, label in zip(img_list[:10], wanted_attrs[:10]):
        # json file creation
        item = {"img_path": img}
        label_ = []
        for i in range(label.size(0)):
            if label[i] == 1:
                label_.append(i)
        item["gt_label"] = label_
        data_list.append(item)

        # create image folders
        img_in_path = os.path.join(dataroot, "img_align_celeba", img)
        img_out_path = os.path.join(output, split, img)
        if not os.path.exists(os.path.join(output, split)):
            os.makedirs(os.path.join(output, split))
        rgb = cv2.imread(img_in_path)
        cv2.imwrite(img_out_path, rgb)

    res["data_list"] = data_list
    res["metainfo"] = meta_info
    return res


def main():
    parser = parse_args()
    data_root = parser.data_root
    output = parser.output
    wanted_labels = [8, 21, 36]  # black_hair, mouth_slightly_opened ,wearing_lipsticks
    output = os.path.join(data_root, output)
    if not os.path.exists(output):
        os.makedirs(output)

    splits = load_csv(data_root, "Anno/list_eval_partition.txt")
    # identity = load_csv(data_root, "Anno/identity_CelebA.txt")
    attr = load_csv(data_root, "Anno/list_attr_celeba.txt", header=1)

    split_map = {"train": 0, "valid": 1, "test": 2}
    for key, value in split_map.items():
        res = convert2dict(splits, attr, data_root, output, wanted_labels, key)
        json_file = os.path.join(output, f"{key}.json")
        with open(json_file, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
