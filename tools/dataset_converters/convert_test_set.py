import argparse
import os
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ImageNet subset lists provided by SimCLR into '
                    'the required format in MMPretrain.')
    parser.add_argument(
        'data_root', help='Input list file, downloaded from SimCLR github repo.')
    parser.add_argument(
        'output', help='Output list file with the required format.')
    parser.add_argument(
        '-m', '--multi_label', help="if use multiple option setting", action='store_true')
    args = parser.parse_args()
    return args


def convert2dict(dataroot: str, parser):
    img_list = os.listdir(dataroot)
    res = {}
    meta_info = {}
    data_list = []
    for img in tqdm(img_list):
        label = [0, 0, 0]  # Fake label creation for custom test set
        # json file creation
        label_ = []
        if parser.multi_label: # generate multiple labels for classification
            item = {"img_path": img}
            item["gt_label"] = label
            data_list.append(item)

        else:  # treat each feature combination as a class
            class_ = 0
            label_.append(class_)
            item = img + " " + str(class_)
            data_list.append(item)

    res["data_list"] = data_list
    res["metainfo"] = meta_info
    return res


def main():
    parser = parse_args()
    data_root = parser.data_root
    output = parser.output
    if not os.path.exists(output):
        os.makedirs(output)

    res = convert2dict(data_root, parser)
    if parser.multi_label:
        json_file = os.path.join(output, f"multi_test.json")
        with open(json_file, "w") as f:
            json.dump(res, f)
        print(f"The list has been saved to {json_file}")
    else:
        txt_file = os.path.join(output, f"test.txt")
        with open(txt_file, 'w') as file:
            # Write each element of the list to the file
            for item in res["data_list"]:
                file.write(str(item) + '\n')
        print(f"The list has been saved to {txt_file}")


if __name__ == "__main__":
    main()
