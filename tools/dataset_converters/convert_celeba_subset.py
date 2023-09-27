import argparse
import csv
import os
from collections import namedtuple
import torch
import numpy as np
import json
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ImageNet subset lists provided by SimCLR into '
        'the required format in MMPretrain.')
    parser.add_argument(
        'data_root', help='Input list file, downloaded from SimCLR github repo.')
    parser.add_argument(
        'output', help='Output list file with the required format.')
    parser.add_argument('-m', '--multi_label', help="if use multiple option setting", action='store_true')
    parser.add_argument('-img', '--save_img', help="if create folder and save them", action='store_true')
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


def convert2dict(splits, attr, dataroot: str, output: str, wanted_labels: list, split: str, wanted_class: list, parser):
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
    for img, label in tqdm(zip(img_list, wanted_attrs)):
        # json file creation
        label_ = []
        if parser.multi_label:
            # generate multiple labels for classification
            item = {"img_path": img}
            for i in range(label.size(0)):
                if label[i] == 1:
                    label_.append(i)
            item["gt_label"] = label_
            class_ = 0
            for i in range(label.size(0)):
                if label[i] == 1:
                    class_ += 2**i
            if class_ in wanted_class:
                data_list.append(item)
        else:
            # treat each feature combination as a class
            class_ = 0
            for i in range(label.size(0)):
                if label[i] == 1:
                    class_ += 2**i
            label_.append(class_)
            item = img + " " + str(class_)
            if class_ in wanted_class:
                data_list.append(item)

        # create image folders
        if parser.save_img:
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
    wanted_class = [1, 2, 3, 5, 6, 7] # remove not blackH not mouth open from dataset
    output = os.path.join(data_root, output)
    if not os.path.exists(output):
        os.makedirs(output)

    splits = load_csv(data_root, "Anno/list_eval_partition.txt")
    # identity = load_csv(data_root, "Anno/identity_CelebA.txt")
    attr = load_csv(data_root, "Anno/list_attr_celeba.txt", header=1)

    split_map = {"train": 0, "valid": 1, "test": 2}
    for key, value in split_map.items():
        res = convert2dict(splits, attr, data_root, output, wanted_labels, key, wanted_class, parser)
        if parser.multi_label:
            json_file = os.path.join(output, f"multi_{key}.json")
            with open(json_file, "w") as f:
                json.dump(res, f)
            print(f"The list has been saved to {json_file}")
        else:
            txt_file = os.path.join(output, f"{key}.txt")
            with open(txt_file, 'w') as file:
                # Write each element of the list to the file
                for item in res["data_list"]:
                    file.write(str(item) + '\n')
            print(f"The list has been saved to {txt_file}")


if __name__ == "__main__":
    main()
