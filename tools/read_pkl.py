import pickle
import csv
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ImageNet subset lists provided by SimCLR into '
                    'the required format in MMPretrain.')
    parser.add_argument(
        'input', help='Input list file, downloaded from SimCLR github repo.')
    parser.add_argument(
        'output', help='Output list file with the required format.')
    args = parser.parse_args()
    return args


def pkl2csv(parser):
    pickle_file = parser.input
    csv_file = parser.output
    try:
        # Load data from the pickle file
        with open(pickle_file, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        new_data = []
        # column_names = data[0].keys()
        num_class = data[0]["num_classes"]
        stats_info_ = np.zeros(2 ** num_class)
        for item in data:
            new_item = {"img_path": item["img_path"]}
            new_item["pred_label"] = item["pred_label"].detach().cpu().numpy()
            new_item["pred_score"] = item["pred_score"].detach().cpu().numpy()
            new_item["gt_label"] = item["gt_label"].detach().cpu().numpy()

            pred_ = new_item["pred_score"] > 0.5
            class_ = 0
            for i in range(num_class):
                class_ += 2 ** i * pred_[i]
            stats_info_[class_] += 1
            new_data.append(new_item)
        stats_info = {}
        for i, value in enumerate(stats_info_):
            stats_info[i] = value

        # Check if the loaded data is a list of dictionaries (common format for CSV)
        if isinstance(new_data, list) and all(isinstance(item, dict) for item in new_data):
            # Extract column names from the first dictionary (assuming all dictionaries have the same keys)

            # Write data to the CSV file
            with open(csv_file, 'w', newline='') as csv_file:
                column_names_stats = stats_info.keys()
                writer_stats = csv.DictWriter(csv_file, fieldnames=column_names_stats)
                writer_stats.writeheader()
                writer_stats.writerow(stats_info)

                column_names = new_data[0].keys()
                writer = csv.DictWriter(csv_file, fieldnames=column_names)
                writer.writeheader()
                for row in new_data:
                    writer.writerow(row)
            print(f"Data from '{pickle_file}' has been successfully saved to '{csv_file}'.")
        else:
            print("Error: The data in the pickle file is not in the expected format (list of dictionaries).")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main():
    parser = parse_args()
    pkl2csv(parser)


if __name__ == "__main__":
    main()
