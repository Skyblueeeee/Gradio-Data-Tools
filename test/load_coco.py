import json
dir = r"Y:\label\label_P-IJC23110092_岚图总装检测\轮胎工位\标注图\发图时间_231219\via_export_coco.json"
with open(dir, 'r',encoding="utf-8") as file:
    # json_data = json.load(file)
    json_data = json.load(file)
    coco_images = json_data["images"]
    coco_ann = json_data["annotations"]

ann_img_id_list =[]
for ann in coco_ann:
    ann_img_id_list.append(int(ann["image_id"]))

for image in coco_images:
    if image["id"] not in ann_img_id_list:
        pass
        # print(image["file_name"])
    
ls=['aa','b','c','ddd','aa']

cou={} #创建一个空字典
for i in ls:
	cou[i]=cou.get(i,0)
print(cou)
