import json
import os

N_txt=0

def coco_to_txt(root, coco_name):
    if coco_name == "":
        return "未输入COCO名称"
    ret_list = []
    if os.path.exists(root):
        for group in os.walk(root):
            for file in group[2]:
                if coco_name in file:
                    via_coco_file = os.path.join(group[0],coco_name+".json")
                    with open(via_coco_file, encoding='UTF-8') as infile:
                        contents = json.load(infile)
                    # print("load file",via_coco_file)
                    anns = contents["annotations"]
                    cats = contents["categories"]
                    imgs = contents["images"]

                    img_path_list = []
                    img_list = []
                    for root, dirs, files in os.walk(group[0]):
                        for name in files:
                            if name[-3:] == 'jpg':
                                img_path_list.append(os.path.join(group[0], name))
                                img_list.append(name)


                    for image_id in range(len(imgs)):
                        img_info_list = []
                        for ann in anns:
                            if ann["bbox"][0] < 0 or ann["bbox"][1] < 0: #防止框飘出图外
                                ann["bbox"][0] = max(ann["bbox"][0], 0)
                                ann["bbox"][1] = max(ann["bbox"][1], 0)
                            if ann["bbox"][2] > imgs[image_id]['width'] or ann["bbox"][3] > imgs[image_id]['height']:
                                ann["bbox"][2] = min(ann["bbox"][2] ,imgs[image_id]['width'])
                                ann["bbox"][3] = min(ann["bbox"][3] ,imgs[image_id]['height'])

                            ann_img_id = int(ann["image_id"])
                            if ann_img_id == image_id:
                                cats_id = ann["category_id"]
                                ann_info_dic = str(cats_id - 1) \
                                            + ' ' + str((ann["bbox"][0] + ann["bbox"][2] / 2) / imgs[image_id]['width']) \
                                            + ' ' + str((ann["bbox"][1] + ann["bbox"][3] / 2) / imgs[image_id]['height']) \
                                            + ' ' + str(ann["bbox"][2] / imgs[image_id]['width']) \
                                            + ' ' + str(ann["bbox"][3] / imgs[image_id]['height']) \
                                            + '\n'
                                img_info_list.append(ann_info_dic)
                        export_name = imgs[image_id]["file_name"]
                        if export_name in img_list:
                            img_index = img_list.index(export_name)
                        else:
                            print("未找到{}此图片".format(export_name))

                        export_file = os.path.join(os.path.split(img_path_list[img_index])[0],
                                                export_name[:-4] + ".txt")
                        with open(export_file, "a+") as out_file:
                            for i in range(len(img_info_list)):
                                out_file.write(img_info_list[i])

                    ret_list.append(f"已完成 {via_coco_file} 的coco2yolo转换")
        res_str = ""
        for res in ret_list:
            res_str += res + "\n\n"
        return  res_str
    else:
        return "未有该路径，请检查！ \n\n暂只支持:    \n\n//10.10.1.125/ai01    \n\n//10.10.1.39/d"

if __name__ == "__main__":
    input_dir=r"\\10.10.1.125\ai01\label\label_P-IJC23110092_岚图总装检测\车门工位\离线增强数据\20231228\From车门工位_发图时间_231204-MJSB标签取出"
    coco_name = "coco_gr"
    # print (coco_to_txt(input_dir,coco_name))

