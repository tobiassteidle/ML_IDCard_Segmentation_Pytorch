import os
import json
import glob

def convert_labelme_to_quad(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    shapes = data.get('shapes', [])
    if not shapes:
        print(f"Skipping {json_path}: No annotations found.")
        return

    points = shapes[0]['points']
    quad_points = [[int(round(x)), int(round(y))] for x, y in points]

    gt_data = {
        "quad": quad_points
    }

    with open(output_path, 'w') as f:
        json.dump(gt_data, f)
    print(f"Converted {json_path} -> {output_path}")

if __name__ == "__main__":
    input_dir = "./labelme_jsons"
    output_dir = "./dataset_labels"
    os.makedirs(output_dir, exist_ok=True)

    for file_path in glob.glob(os.path.join(input_dir, "*.json")):
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_dir, filename)
        convert_labelme_to_quad(file_path, out_path)
