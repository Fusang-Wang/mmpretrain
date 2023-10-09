import json
import random
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ImageNet subset lists provided by SimCLR into '
                    'the required format in MMPretrain.')
    parser.add_argument(
        'input', help='Input json file, downloaded from SimCLR github repo.')
    parser.add_argument(
        'output', help='Output json file with the required format.')
    parser.add_argument(
        '-max', help='maximum number of img for output dataset', default=1000)
    args = parser.parse_args()
    return args


def select_and_save_random_images(input_file, output_file, num_images_to_select):
    try:
        # Read the JSON file
        with open(input_file, 'r') as json_file:
            data = json.load(json_file)

        # Shuffle the list of image data
        new_data = {'metainfo': data["metainfo"]}
        datalist = data["data_list"]
        random.shuffle(datalist)

        # Select the specified number of images
        selected_datalist = datalist[:num_images_to_select]
        new_data["data_list"] = selected_datalist

        # Write the selected images to a new JSON file
        with open(output_file, 'w') as new_json_file:
            json.dump(new_data, new_json_file, indent=4)

        print(f"{num_images_to_select} random images selected and saved to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
# Replace 'input.json', 'output.json', and 'num_images_to_select' with your file paths and desired number


def main():
    parser = parse_args()
    input_json = parser.input
    ouput_json = parser.output
    num_max = int(parser.max)
    select_and_save_random_images(input_json, ouput_json, num_max)

if __name__ == "__main__":
    main()
