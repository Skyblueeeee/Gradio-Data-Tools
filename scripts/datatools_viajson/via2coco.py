import os, json

def init_coco(categories: list, supercategory: str):
    coco = {
        "info": {
            "year": 2023,
            "version": "1",
            "description": "Exported from python",
            "contributor": "Skyblueee",
            "url": "Skyblueeeee@github",
            "date_created": "no need record"
        },
        "images": [],
        "annotations": [],
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": []
    }
    for k, v in categories.items():
        coco["categories"].append({
            "id": findlabelnum(k,categories),
            "name": k,
            "supercategory": supercategory
        })
    return coco
def findlabelnum(v,dic):
    for i,key in enumerate(dic):
        if v == key:
            return i+1
    assert -1
N_COCO_ID = 0
def run_viatococo(root_dir,via_name, save_coco_name,supercategory,input_img_width=None,input_img_heigh=None):
    #########################注意事项#########################
    # 该脚本不会转换无效图和其上的标签
    #########################参数调整#########################
    # 如果为None，则保存到同名文件夹, 否则保存到固定路径下
    #########################################################
    if save_coco_name == "":
        return "未输入COCO名称"
    if supercategory == "":
        supercategory="fitow"
    if via_name == "":
        via_name = "via_project"
    ret_list = []
    if os.path.exists(root_dir):
        for group in os.walk(root_dir):
            for each_file in group[2]:
                if each_file.endswith(".json") and via_name+".json" in each_file:
                    # 初始化coco
                    each_json = os.path.join(group[0], each_file)
                    with open(each_json, "r", encoding="utf-8") as fp:
                        via_dict = json.load(fp)
                    supercate = list(via_dict["_via_attributes"]["region"].keys())[0]
                    if supercate != supercategory:
                        return f"模板名称不一致!请检查标注文件模板!\n\n文件:{supercate} 输入:{supercategory}"

                    cat_dict = via_dict["_via_attributes"]["region"][supercategory]["options"]
                    coco_dict = init_coco(cat_dict,supercategory)
                    N_imgs = 0
                    N_anns = 0
                    coco_img_dict = coco_dict["images"]
                    coco_ann_dict = coco_dict["annotations"]
                    # 开始转换
                    for each_image_info in via_dict["_via_img_metadata"].values():
                        if "width" in each_image_info:
                            img_width = each_image_info["width"]
                            img_height = each_image_info["height"]
                        else:
                            file_path = os.path.join(group[0],each_image_info["filename"])
                            if os.path.exists(file_path):
                                from PIL import Image
                                img = Image.open(file_path)
                                img_width,img_height = img.size
                            else:
                                img_width,img_height = input_img_width,input_img_heigh
                        each_file_attr = each_image_info["file_attributes"].get("fileattr", "有效")
                        if each_file_attr not in ["有效", "空图"]:
                            continue
                        coco_img_dict.append({
                            "id": N_imgs,
                            "width": img_width,
                            "height": img_height,
                            "file_name": each_image_info["filename"],
                            "license": 1,
                            "date_captured": ""
                        })
                        for each_via_ann in each_image_info["regions"]:
                            s_a = each_via_ann["shape_attributes"]
                            b_x, b_y, b_w, b_h = s_a["x"], s_a["y"], s_a["width"], s_a["height"]
                            coco_ann_dict.append({
                                "id": N_anns,
                                "image_id": N_imgs,
                                "category_id": findlabelnum(each_via_ann["region_attributes"][supercategory],cat_dict),
                                "segmentation": [[
                                    b_x, b_y,
                                    b_x + b_w, b_y,
                                    b_x + b_w, b_y + b_h,
                                    b_x, b_y + b_h,
                                ]],
                                "area": b_w * b_h,
                                "bbox": [b_x, b_y, b_w, b_h],
                                "iscrowd": 0
                            })
                            N_anns += 1
                        N_imgs += 1
                    # 开始存储
                    save_coco_json_path_1 = os.path.join(group[0], save_coco_name + ".json")
                    with open(save_coco_json_path_1, "w", encoding="utf-8") as fp:
                        json.dump(coco_dict, fp, ensure_ascii=False)
                    ret_list.append(f"已完成 {save_coco_json_path_1} 的viatococo转换")
        res_str = ""
        for res in ret_list:
            res_str += res + "\n\n"
        return  res_str
                    # os.makedirs(save_dir_2, exist_ok=True)
                    # save_coco_json_path_2 = os.path.join(save_dir_2, save_coco_name + '_' + str(N_COCO_ID) + ".json")
                    # with open(save_coco_json_path_2, "w", encoding="utf-8") as fp:
                    #     json.dump(coco_dict, fp, ensure_ascii=False)
                    # print("从{}转到{}已存储：\n共发现{}张图，实际转换图：{}张，标签数：{}个\n".format(
                    #     each_json,
                    #     os.path.basename(save_coco_json_path_2),
                    #     len(via_dict["_via_img_metadata"].keys()),
                    #     N_imgs - 1,
                    #     N_anns - 1
                    # ))
    
    else:
        return "未有该路径,请检查! \n\n暂只支持:    \n\n//10.10.1.125/ai01    \n\n eg2.//10.10.1.39/d"
    
# root_dir=r"Y:\label\label_P-IJC23110092_岚图总装检测\车门工位\离线增强数据\20231228\From车门工位_发图时间_231204-MJSB标签取出"
# save_coco_name="coco_test"
# run_viatococo(root_dir,save_coco_name,supercategory="default")