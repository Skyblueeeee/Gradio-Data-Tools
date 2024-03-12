 # -*- encoding: utf-8 -*-
import json
import os
import numpy as np
import cv2

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

def run_yolo2via(data_dir,via_name,super_cate,options):
    label_dict = {}
    for group in options.to_dict("split")["data"]:
        if group[0]!= "":
            label_dict[group[0]]=group[1]
    category_id = {i+1: key for i, key in enumerate(label_dict)}
    via_json = {
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
                    "region_label": "default",  
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
                    "options": options,
                    "default_options": { "": "true" }
                }
                },
                "file": {
                    "fileattr": {
                    "type": "dropdown",
                    "description": "",
                    "options": {
                        "有效": "",
                        "无效": "",
                        "过滤": ""
                    },
                    "default_options": {
                        "有效": True
                    }
                }
                }
            }
    }
    n = 0
    for root, dirs, files in os.walk(data_dir):
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
                via_json["_via_img_metadata"][img_metadata_key] = img_metadata
                # print(key_file_names)
                n += 1

    projest_labels = {}
    for label_index, single_category in category_id.items():
        projest_labels[single_category] = ""
    via_json["_via_attributes"]["region"][super_cate]["options"] = projest_labels
    with open(data_dir + "\\" + via_name + '.json', 'w',encoding="utf-8") as f:
        json.dump(via_json, f,ensure_ascii=False)
    return f"已完成 {data_dir}  yolo2via的转换 imgs_nums:{n}"

if __name__ == "__main__":
    yolo_data_dir = r'D:\20220819'

    options = {     "ck":"",
                    "wh":"",
                    "yb":""
                }

    super_cate = "default"
    run_yolo2via(yolo_data_dir,options,super_cate)

