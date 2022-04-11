import json
import os

from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import shutil 

parser = argparse.ArgumentParser()
parser.add_argument('-s',
                    '--set',
                    help='choose between train, validation and test',
                    choices=['train', 'validation', 'test', 'train2', 'validation2', 'train2_scale512', 'validation2_scale512'],
                    nargs='?',
                    default='validation')
parser.add_argument('-d',
                    '--dest',
                    help='output dir',
                    type=str,
                    default=os.path.join('./data', 'deepfashion2_cat1'))
args = parser.parse_args()

dataset = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

lst_name = ['short_sleeved_shirt']

for idx, e in enumerate(lst_name):
    dataset['categories'].append({
        'id': idx + 1,
        'name': e,
        'supercategory': "clothes",
        'keypoints': ['%i' % i for i in range(1, 25)],
        'skeleton': [[1,2],[2,3],[3,4],[4,5],[5,6],[2,7],[7,8],[8,9],[9,10],[10,11],
                    [11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,19],
                    [19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,6]]
    })

all_files = os.listdir(os.path.join('./src/data/deepfashion2', args.set, 'image'))
sub_index = 0  # the index of ground truth instance

for file in tqdm(all_files):
    json_name = os.path.join('./data/deepfashion2', args.set, 'annos', os.path.splitext(file)[0] + '.json')
    image_name = os.path.join('./data/deepfashion2', args.set, 'image', file)
    dest_image_name = os.path.join('./data', 'deepfashion2_cat1', 'images', args.set , file)

    if int(os.path.splitext(file)[0]) >= 0:
        imag = Image.open(image_name)
        width, height = imag.size
        with open(json_name, 'r') as f:
            temp = json.loads(f.read())
            pair_id = temp['pair_id']

            for i in temp:
                if i == 'source' or i == 'pair_id':
                    continue
                elif temp[i]['category_id'] ==1:
                    points = np.zeros(25 * 3)
                    sub_index = sub_index + 1
                    box = temp[i]['bounding_box']
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    x_1 = box[0]
                    y_1 = box[1]
                    bbox = [x_1, y_1, w, h]
                    cat = temp[i]['category_id']
                    style = temp[i]['style']
                    seg = temp[i]['segmentation']
                    landmarks = temp[i]['landmarks']


                    points_x = landmarks[0::3]
                    points_y = landmarks[1::3]
                    points_v = landmarks[2::3]
                    points_x = np.array(points_x)
                    points_y = np.array(points_y)
                    points_v = np.array(points_v)

                    case = [0, 25, 58, 89, 128, 143, 158, 168, 182, 190, 219, 256, 275, 294]
                    idx_i, idx_j = case[cat - 1], case[cat]

                    for n in range(idx_i, idx_j):
                        if len(points_x) > n:
                            points[3 * n] = points_x[n - idx_i]
                            points[3 * n + 1] = points_y[n - idx_i]
                            points[3 * n + 2] = points_v[n - idx_i]
                        else:
                            points[3 * n] = 0
                            points[3 * n + 1] = 0
                            points[3 * n + 2] = 0
                    num_points = len(np.where(points_v > 0)[0])

                    dataset['annotations'].append({
                        'area': w * h,
                        'bbox': bbox,
                        'category_id': cat,
                        'id': sub_index,
                        'pair_id': pair_id,
                        'image_id': int(os.path.splitext(file)[0]),
                        'iscrowd': 0,
                        'style': style,
                        'num_keypoints': num_points,
                        'keypoints': points.tolist(),
                        'segmentation': seg,
                    })

                    dataset['images'].append({
                        'coco_url': '',
                        'date_captured': '',
                        'file_name': file,
                        'flickr_url': '',
                        'id': int(os.path.splitext(file)[0]),
                        'license': 0,
                        'width': width,
                        'height': height
                    })

                    shutil.copyfile(image_name, dest_image_name) #copy image file to other folder


json_name = os.path.join(args.dest, person_keypoints_' + args.set + '.json')
with open(json_name, 'w') as f:
    json.dump(dataset, f, indent=4)
