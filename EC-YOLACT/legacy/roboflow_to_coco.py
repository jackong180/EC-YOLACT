import os
import glob
import json
import random
import shutil

from PIL import Image
from pycocotools import mask as maskUtils
import numpy as np

#==========================
# 修改下面三个路径
#==========================

input_dir = "datasets/before"

img_out = "datasets/map_out/JPEGImages"

json_out = "datasets/map_out/Jsons"

classes_path = "model_data/cls_classes.txt"

train_percent = 0

os.makedirs(img_out, exist_ok=True)
os.makedirs(json_out, exist_ok=True)

#########################################################

with open(classes_path, encoding='utf-8') as f:
    class_names = [x.strip() for x in f.readlines()]

class2id = {}

for i, cls in enumerate(class_names):
    class2id[cls] = i + 1

#########################################################

json_files = glob.glob(os.path.join(input_dir, "*.json"))

random.seed(0)
random.shuffle(json_files)

num_train = int(len(json_files) * train_percent)

train_files = json_files[:num_train]
val_files = json_files[num_train:]


def create_empty():

    return {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": class2id[c],
                "name": c,
                "supercategory": c
            }
            for c in class_names
        ]
    }


train_json = create_empty()
val_json = create_empty()

ann_id = 0


def convert(file_list, coco):

    global ann_id

    for image_id, jf in enumerate(file_list):

        print(jf)

        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        img_name = data["annotation_data"]["key"]

        img_path = os.path.join(input_dir, img_name)

        if not os.path.exists(img_path):
            print(img_path, "不存在")
            continue

        shutil.copy(img_path,
                    os.path.join(img_out, img_name))

        img = Image.open(img_path)

        w, h = img.size

        coco["images"].append({

            "id": image_id,

            "file_name": img_name,

            "width": w,

            "height": h

        })

        boxes = data["annotation_data"]["boxes"]

        for obj in boxes:

            label = obj["label"]

            if label not in class2id:
                continue

            pts = np.array(obj["points"], dtype=np.float32)

            segmentation = [pts.reshape(-1).tolist()]

            x = pts[:, 0]
            y = pts[:, 1]

            xmin = float(np.min(x))
            xmax = float(np.max(x))

            ymin = float(np.min(y))
            ymax = float(np.max(y))

            bbox = [

                xmin,

                ymin,

                xmax - xmin,

                ymax - ymin

            ]

            mask = np.zeros((h, w), np.uint8)

            from PIL import ImageDraw

            mask_img = Image.fromarray(mask)

            draw = ImageDraw.Draw(mask_img)

            draw.polygon([tuple(p) for p in pts],
                         outline=1,
                         fill=1)

            mask = np.array(mask_img)

            rle = maskUtils.encode(
                np.asfortranarray(mask)
            )

            area = float(maskUtils.area(rle))

            coco["annotations"].append({

                "id": ann_id,

                "image_id": image_id,

                "category_id": class2id[label],

                "segmentation": segmentation,

                "area": area,

                "bbox": bbox,

                "iscrowd": 0

            })

            ann_id += 1


convert(train_files, train_json)
convert(val_files, val_json)

with open(os.path.join(json_out,
                       "train_annotations.json"),
          "w",
          encoding="utf-8") as f:

    json.dump(train_json,
              f,
              indent=4)

with open(os.path.join(json_out,
                       "val_annotations.json"),
          "w",
          encoding="utf-8") as f:

    json.dump(val_json,
              f,
              indent=4)

print()
print("===================================")
print("转换完成！")
print("train =", len(train_json["images"]))
print("val   =", len(val_json["images"]))
print("===================================")