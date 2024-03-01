import json
import os 

def main_coco2via(img_path,coco_name,via_name,moudel_name):
    if coco_name == "":
        return "未输入COCO名称"
    if moudel_name == "":
        moudel_name="fitow"
    ret_list = []
    if os.path.exists(img_path):
        for group in os.walk(img_path):
            for each_file in group[2]:
                if each_file.endswith(".json") and coco_name+".json" in each_file:
                    coco_path = os.path.join(group[0],coco_name+".json")
                    save_path = os.path.join(group[0],via_name+".json")
                    imageIdName = {}
                    imageWName = {}
                    imageHName = {}
                    classIdName = {}
                    via_img_metadata = {}
                    options = {}
                    regions = []

                    with open(coco_path, 'r', encoding='utf-8') as f :
                        coco_data = json.load(f)

                    for image_info in coco_data['images'] :
                        imageIdName[image_info['id']] = image_info['file_name']

                    for image_info in coco_data['images'] :
                        imageWName[image_info['id']] = image_info['width']

                    for image_info in coco_data['images'] :
                        imageHName[image_info['id']] = image_info['height']

                    for class_info in coco_data['categories'] :
                        classIdName[class_info['id']] = class_info['name']

                    for num, cla in classIdName.items() :
                        options[cla] = ''

                    via_attributes = {
                        "region": {
                        moudel_name: {
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

                    via_settings = {
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
                            "region_label": "fitow",
                            "region_color": "fitow",
                            "region_label_font": "10px Sans",
                            "on_image_annotation_editor_placement": "NEAR_REGION"
                        }
                        },
                        "core": {
                        "buffer_size": 18,
                        "filepath": {},
                        "default_filepath": ""
                        },
                        "project": { "name": "via_project" }
                    }
                    
                    oldID = -1
                    for ann in coco_data['annotations'] :
                        nowID = ann['image_id']
                        if nowID != oldID :
                            for num_null in range(int(nowID) - int(oldID) - 1, -1 , -1) :
                                if num_null != 0 :
                                    # print(num_null)
                                    pass
                                filename = imageIdName[int(nowID) - num_null]
                                width = imageWName[int(nowID) - num_null]
                                height = imageHName[int(nowID) - num_null]
                                imageSize = os.path.getsize(os.path.join(group[0], filename))
                                viaDataName = filename + str(imageSize)
                                via_img_metadata[viaDataName] = {
                                        "filename": filename,
                                        "width":width,
                                        "height":height,
                                        "size": imageSize,
                                        "regions": [],
                                        "file_attributes": {}
                                        }
                        oldID = nowID
                        filename = imageIdName[int(nowID) - num_null]
                        imageSize = os.path.getsize(os.path.join(group[0], filename))
                        viaDataName = filename + str(imageSize)
                        width = imageWName[int(nowID) - num_null]
                        height = imageHName[int(nowID) - num_null]
                        imageSize = os.path.getsize(os.path.join(group[0], filename))
                        attribute = {
                            "shape_attributes": {
                                "name": "rect",
                                "x": ann['bbox'][0],
                                "y": ann['bbox'][1],
                                "width": ann['bbox'][2],
                                "height": ann['bbox'][3]
                            },
                            "region_attributes": { moudel_name: classIdName[ann['category_id']] }
                            }
                        via_img_metadata[viaDataName]['regions'].append(attribute)

                    via = {
                        '_via_settings' : via_settings,
                        '_via_img_metadata' : via_img_metadata,
                        '_via_attributes' : via_attributes
                    }

                    with open(save_path, 'w', encoding='utf_8') as f :
                        json.dump(via, f,ensure_ascii=False)

                    ret_list.append(f"已完成 {save_path} 的coco2via转换")
        res_str = ""
        for res in ret_list:
            res_str += res + "\n\n"
        return  res_str
    else:
        return "未有该路径,请检查! \n\n暂只支持:    \n\n//10.10.1.125/ai01    \n\n eg2.//10.10.1.39/d"

# img_path=r"Y:\label\label_P-IJC23110092_岚图总装检测\车门工位\离线增强数据\20231228\From车门工位_发图时间_231204-MJSB标签取出"
# coco_name="coco_test"
# via_name = "via_project_test.py"
# moudel_name="default"
# main_coco2via(img_path,coco_name,via_name,moudel_name)