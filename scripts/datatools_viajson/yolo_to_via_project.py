 # -*- encoding: utf-8 -*-
import json
import os
import numpy as np
import cv2

yolo_data_dir = r''

category_id = { 1:"ck",
                2:"wh",
                3:"yb"
               }

super_cate = "default"

# 建立一个VIA格式的模板
json_ = {
    "_via_settings": {
        "ui": {
            "annotation_editor_height": 25,
            "annotation_editor_fontsize": 0.8,
            "leftsidebar_width": 18,
            "image_grid": {
                "img_height": 80,
                "rshape_fill": "none",
                "rshape_fill_opacity": 0.3,
                "rshape_stroke": "yellow",
                "rshape_stroke_width": 2,
                "show_region_shape": True,
                "show_image_policy": "all"
            },
            "image": {
                "region_label": super_cate,  # 这里如果无超类，那这里就是类的代号
                "region_color": "__via_default_region_color__",
                "region_label_font": "10px Sans",
                "on_image_annotation_editor_placement": "NEAR_REGION"
            }
        },
        "core": {
            "buffer_size": 18,
            "filepath": {},
            "default_filepath": ""
        },
        "project": {"name": "pre_prediction"}  # 这里是VIA文件的名字
    },
    "_via_img_metadata": {},
    "_via_attributes": {
        "region": {
            super_cate: {
                "type": "dropdown",
                "description": "",
                # “_via_attributes”下的"options"是每个类别的空值字典，照着填好即可
                "options": {
                    "lw": "",
                    "kp": "",
                    "sxyw": "",
                    "ql": "",
                    "ttmb": ""

                },
                "default_options": {}
            }
        },
        "file": {}
    }
}


def extract_yololabel_inf(root, key_file_names, img_w, img_h):
    with open(root + '/' + key_file_names[:-3] + 'txt', "r") as f:
        img_label_inf = {}
        key_imgname = str(key_file_names)
        img_label_inf[key_imgname] = []
        line_list = []
        lines = f.readlines()
        for line in lines:
            line_inf = line.split()
            line_list.append(int(line_inf[0]) + 1)
            line_list.append(float(line_inf[1]) * img_w - float(line_inf[3]) * img_w / 2)
            line_list.append(float(line_inf[2]) * img_h - float(line_inf[4]) * img_h / 2)
            # line_list.append(float(line_inf[1]) * img_w + float(line_inf[3]) * img_w / 2)
            # line_list.append(float(line_inf[2]) * img_h + float(line_inf[4]) * img_h / 2)
            line_list.append(float(line_inf[3]) * img_w)
            line_list.append(float(line_inf[4]) * img_h)
            img_label_inf[key_imgname].append(line_list)
            line_list = []
        return img_label_inf

n = 0
for root, dirs, files in os.walk(yolo_data_dir):
    for key_file_names in files:
        if key_file_names[-3:] == 'jpg':
            img_wh = cv2.imdecode(np.fromfile(os.path.join(root, key_file_names), dtype=np.uint8), 1)
            img_w = img_wh.shape[1]
            img_h = img_wh.shape[0]
            img_label_inf = extract_yololabel_inf(root, key_file_names, img_w, img_h)
            value_DT = img_label_inf[key_file_names]
            img_metadata = {
                "filename": "",
                "size": None,
                "regions": [],
                "file_attributes": {}
            }
            file_name = key_file_names
            img_path = os.path.join(root, file_name)
            img = open(img_path, 'rb').read()

            size = len(img)
            for each_DT in value_DT:  # 获取DT类别id，DTbbox
                # 建立结构体
                region_dict = {
                    "shape_attributes": {
                        "name": "rect",
                        "x": None,
                        "y": None,
                        "width": None,
                        "height": None
                    },
                    "region_attributes": {super_cate: None}
                }

                category = category_id[each_DT[0]]
                w = each_DT[3]
                h = each_DT[4]
                x = each_DT[1]
                y = each_DT[2]

                region_dict["region_attributes"][super_cate] = category
                region_dict["shape_attributes"]["x"] = x
                region_dict["shape_attributes"]["y"] = y
                region_dict["shape_attributes"]["width"] = w
                region_dict["shape_attributes"]["height"] = h
                img_metadata["regions"].append(region_dict)
            img_metadata["filename"] = file_name
            img_metadata["size"] = size
            img_metadata_key = file_name + str(size)
            json_["_via_img_metadata"][img_metadata_key] = img_metadata
            print(key_file_names)
            n += 1
            print(n)

projest_labels = {}
for label_index, single_category in category_id.items():
    projest_labels[single_category] = ""
json_["_via_attributes"]["region"][super_cate]["options"] = projest_labels
with open(yolo_data_dir + '\\via_project.json', 'w') as f:
    json.dump(json_, f)
print("DONE")
