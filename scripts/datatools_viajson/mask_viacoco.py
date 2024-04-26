import json
import datetime
from PIL import Image
import os

via_data = {
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
        "project": {
            "name": "via_project"
        }
    },
    "_via_img_metadata": {},
    "_via_attributes": {
        "region": {},
        "file": {}
    }
}

def get_image_size(file_path):
    with Image.open(file_path) as img:
        return img.size

def convertMASK_COCO2VIA(img_dir, via_name, coco_name, supercategory,input_img_width=None,input_img_heigh=None):
    if coco_name == "":
        return "未输入COCO名称"
    if supercategory == "":
        supercategory="fitow"
    if via_name == "":
        via_name = "via_project"
    ret_list = []
    if os.path.exists(img_dir):
        for group in os.walk(img_dir):
            for each_file in group[2]:
                if each_file.endswith(".json") and coco_name+".json" in each_file:
                    via_path = os.path.join(img_dir,via_name + ".json")
                    coco_path = os.path.join(img_dir, each_file)

                    with open(coco_path, 'r', encoding='utf-8') as file:
                        coco_data = json.load(file)

                    # 构建VIA的attributes
                    categories = coco_data.get("categories", [])
                    for category in categories:
                        category_name = category['name']
                        # supercategory = category['supercategory']
                        via_data["_via_attributes"]["region"][supercategory] = {
                            "type": "dropdown",
                            "description": "",
                            "options": {
                                category_name: ""
                            },
                            "default_value": category_name
                        }

                    images = coco_data.get("images", [])
                    annotations = coco_data.get("annotations", [])

                    for img in images:
                        img_filename = img['file_name']
                        img_id = img['id']
                        img_path = os.path.join(img_dir, img_filename)
                        img_width, img_height = get_image_size(img_path)
                        img_size = os.path.getsize(img_path)
                        via_data["_via_img_metadata"][img_filename+str(img_size)] = {
                            "filename": img_filename,
                            "width":img_width,
                            "height":img_height,
                            "size": img_size,
                            "regions": []
                        }

                    for annotation in annotations:
                        img_id = annotation['image_id']
                        file_key = images[img_id]['file_name'] + str(os.path.getsize(os.path.join(img_dir,images[img_id]['file_name'])))  # COCO的image_id是1-based
                        region = via_data["_via_img_metadata"][file_key]
                        category_id = annotation['category_id']
                        category_name = coco_data["categories"][category_id - 1]['name']
                        segmentation = annotation['segmentation'][0]

                        region['regions'].append({
                            "shape_attributes": {
                                "name": "polygon",
                                "all_points_x": [point for point in segmentation[::2]],
                                "all_points_y": [point for point in segmentation[1::2]]
                            },
                            "region_attributes": {
                                supercategory: category_name
                            }
                        })

                    # 写入VIA格式的JSON数据
                    with open(via_path, 'w', encoding='utf-8') as file:
                        json.dump(via_data, file, ensure_ascii=False, indent=4)
                    ret_list.append(f"已完成 {via_path} 的coco2via转换")

        res_str = ""
        for res in ret_list:
            res_str += res + "\n\n"
        return  res_str
    
    else:
        return "未有该路径,请检查! \n\n暂只支持:    \n\n//10.10.1.125/ai01    \n\n eg2.//10.10.1.39/d"
    

def convertMASK_VIA2COCO(img_dir, via_name,coco_name, supercategory,input_img_width=None,input_img_heigh=None):
    if coco_name == "":
        return "未输入COCO名称"
    if supercategory == "":
        supercategory="fitow"
    if via_name == "":
        via_name = "via_project"
    ret_list = []
    if os.path.exists(img_dir):
        for group in os.walk(img_dir):
            for each_file in group[2]:
                if each_file.endswith(".json") and via_name+".json" in each_file:
                    via_path = os.path.join(group[0], each_file)
                    coco_path = os.path.join(img_dir, coco_name+ ".json")

                    now_datetime = datetime.datetime.now()
                    images_list = []
                    annotations_list = []
                    categories_list = []
                    with open(via_path, 'r', encoding='utf-8') as f:
                        project_dict = json.load(f)
                        # supercategory = list(project_dict['_via_attributes']['region'].keys())[0]
                        options_list = list(list(project_dict['_via_attributes']['region'].values())[0]['options'].keys())
                        for i in range(len(options_list)):
                            each_category_dict = {
                                'id': i + 1,
                                'name': options_list[i],
                                'supercategory': supercategory
                            }
                            categories_list.append(each_category_dict)
                        image_id = 0
                        annotation_id = 0
                        for each_val in project_dict['_via_img_metadata'].values():
                            # print(f"正在处理第{image_id + 1}张图片： {each_val['filename']}")
                            image_path = os.path.join(img_dir, each_val['filename'])
                            try:
                                img = Image.open(image_path)
                            except:
                                # TODO 使用日志系统，替换print
                                print(f'图片不存在或者无法读取，图片路径为{image_path}')
                                raise Exception('读取图片发生错误。')

                            w, h = img.size
                            each_image_dict = {
                                'id': image_id,
                                'width': w,
                                'height': h,
                                'file_name': each_val['filename'],
                                'license': 1,
                                'date_captured': ''
                            }
                            images_list.append(each_image_dict)

                            for each_region in each_val['regions']:
                                if 'x' in each_region['shape_attributes']:
                                    x = each_region['shape_attributes']['x']
                                    y = each_region['shape_attributes']['y']
                                    width = each_region['shape_attributes']['width']
                                    height = each_region['shape_attributes']['height']
                                    segList = [x, y, x + width, y, x + width, y + height, x, y + height]
                                else:
                                    all_points_x_list = each_region['shape_attributes']['all_points_x']
                                    all_points_y_list = each_region['shape_attributes']['all_points_y']
                                    segList = []
                                    for x, y in zip(all_points_x_list, all_points_y_list):
                                        segList.append(x)
                                        segList.append(y)
                                    x = min(all_points_x_list)
                                    y = min(all_points_y_list)
                                    width = max(all_points_x_list) - x
                                    height = max(all_points_y_list) - y
                                    if width <= 0 or height <= 0:
                                        # TODO 使用日志系统，替换print
                                        print('异常的标注框：')
                                        print(each_val['filename'])
                                        print(all_points_x_list)

                                each_annotation_dict = {
                                    'id': annotation_id,
                                    'image_id': image_id,
                                    'category_id': options_list.index(each_region['region_attributes'][supercategory]) + 1,
                                    # TODO segmentation为bbox格式，如果使用mask需要修改segmentation
                                    'segmentation': [segList],
                                    'area': width * height,
                                    'bbox': [x, y, width, height],
                                    'iscrowd': 0  # TODO iscrow目前固定为0
                                }

                                annotations_list.append(each_annotation_dict)
                                annotation_id += 1
                            image_id += 1

                    via_export_coco_dict = {
                        'info': {
                            'year': int(now_datetime.year),
                            'version': '1',
                            'description': 'Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)',
                            'contributor': '',
                            'url': 'http://www.robots.ox.ac.uk/~vgg/software/via/',
                            'date_created': now_datetime.strftime('%a %b %d %Y %H:%M:%S GMT+0800')
                        },
                        'images': images_list,
                        'annotations': annotations_list,
                        'licenses': [{
                            'id': 1,
                            'name': 'Unknown',
                            'url': ''
                        }],
                        'categories': categories_list
                    }

                    with open(coco_path, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(via_export_coco_dict, ensure_ascii=False))
                    ret_list.append(f"已完成 {coco_path} 的via2coco转换")

        res_str = ""
        for res in ret_list:
            res_str += res + "\n\n"
        return  res_str

    else:
        return "未有该路径,请检查! \n\n暂只支持:    \n\n//10.10.1.125/ai01    \n\n eg2.//10.10.1.39/d"

if __name__ == "__main__":
    img_dir = r""
    coco_file = "coco"
    via_file = "via"
    supercategory = "default"

    print(convertMASK_COCO2VIA(img_dir,via_file,coco_file, supercategory))
    # convertMASK_VIA2COCO(img_dir,via_file,coco_file, super_region)