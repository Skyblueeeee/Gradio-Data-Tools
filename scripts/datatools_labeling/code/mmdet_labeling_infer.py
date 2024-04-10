import cv2
import os, json, time
import numpy as np
from PIL import Image
from mmdet.apis import inference_detector, init_detector
from scripts.datatools_viajson.via2coco import run_viatococo

def file_info(dir):
    ck = None
    cf = None
    for file in os.listdir(dir):
        if file.endswith(".pth"):
            ck = os.path.join(dir,file)
        elif file.endswith(".py"):
            cf = os.path.join(dir,file)
    # print(ck,cf)
    return ck,cf
def read_images(image_path):
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image

dirs = r"scripts\datatools_labeling\models\lantu_yb_hec"
checkpoint,config = file_info(dirs)

class MM_Infer_Labeling():
    def __init__(self) -> None:
        self.super_category = "default"
        self.via_name = "via_infer"
        # self.model_init(config=config,checkpoint=checkpoint,device="cuda:1")
    def pid_init(self):
        pidd = "log/pid"
        if not os.path.isdir(pidd):
            os.makedirs(pidd)
        # if len(os.listdir(pidd)) == 0:
        with open(f"log/pid/{os.getpid()}", "w") as fp:
            fp.write("")

    def json_init(self):
        pro_json = {
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
                        "region_label": "__via_region_id__",
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
                "project": {
                    "name": "via_project"
                }
            },
            "_via_img_metadata": {},
            "_via_attributes": {
                "region": {
                    self.super_category: {
                        "type": "dropdown",
                        "description": "",
                        "options": {},
                        "default_options":{}
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
        return pro_json

    def draw_boxes(self,image, labels):
        for label, info in labels.items():
            threshold, x, y, w, h = info
            x, y, w, h = int(x), int(y), int(w), int(h)
            color = (255, 0, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {threshold:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def model_infer_sever(self,config,checkpoint,device,image,score_th):
        res,res_img = self.model_single_infer(image,score_th)

    def model_init(self,config,checkpoint,device):
        self.model = init_detector(config, checkpoint, device=device)

    def model_infer(self,image_dir,score_thr,outfilemode,filter_labels=[]):
        project_save_path = os.path.join(image_dir,self.via_name+".json")

        pro_json = self.json_init()
        via_images = pro_json["_via_img_metadata"]
        via_attrs = pro_json["_via_attributes"]["region"][self.super_category]["options"]

        for each_label in self.model.CLASSES:
            via_attrs[each_label] = ""

        n_img = 0
        for each_jpg in os.listdir(image_dir):
            if each_jpg.endswith((".jpg", ".png")):
                t1 = time.time()
                image_path = os.path.join(image_dir, each_jpg)

                regions = []
                result = inference_detector(self.model, image_path)
                for i in range(len(result)):
                    class_name = self.model.CLASSES[i]
                    if class_name in filter_labels:
                        continue
                    else:
                        for each_bbox_info in result[i]:
                            x1, y1, x2, y2, c = each_bbox_info.tolist()
                            if c < score_thr:
                                continue
                            regions.append({
                                "shape_attributes": {
                                    "name": "rect",
                                    "x": round(x1, 2),
                                    "y": round(y1, 2),
                                    "width": round(x2 - x1, 2),
                                    "height": round(y2 - y1, 2)
                                },
                                "region_attributes": {
                                    self.super_category: class_name
                                }
                            })

                img_size = os.path.getsize(image_path)
                img = Image.open(image_path)
                width,height = img.size
                via_images[each_jpg + str(img_size)] = {
                    "filename": each_jpg,
                    "width":width,
                    "height":height,
                    "size": img_size,
                    "regions": regions,
                    "file_attributes": {}
                }
                n_img += 1
                t2 = time.time()
                # print(f"{n_img}、{each_jpg}添加完毕! 构建耗时{(t2 - t1)*1000:.2f}ms")

        if outfilemode == "ALL":
            with open(os.path.join(project_save_path), "w", encoding="utf-8") as fp:
                json.dump(pro_json, fp, ensure_ascii=False, indent=4)
            run_viatococo(image_dir,self.via_name,"coco_label",self.super_category)
        else:
            with open(os.path.join(project_save_path), "w", encoding="utf-8") as fp:
                json.dump(pro_json, fp, ensure_ascii=False, indent=4)

    def model_single_infer(self,imgs,score_th):
        res = {}
        result = inference_detector(self.model, imgs)
        for i in range(len(result)):
            class_name = self.model.CLASSES[i]
            for each_bbox_info in result[i]:
                x1, y1, x2, y2, score = each_bbox_info.tolist() 
                if score >= score_th:
                    res[class_name] = [score,round(x1, 2),round(y1, 2),round(x2 - x1, 2), round(y2 - y1, 2)]
        res_img = self.draw_boxes(imgs,res)
        return res,res_img
    
if __name__ == "__main__":
    image_dir = r"D:\test\仪表数据\仪表台_20231127_HJK_XRVALPAHJKAKC00000_LDP95H96XPE310436_01_1.jpg"
    image = read_images(image_dir)
    mminfer = MM_Infer_Labeling()
    res,res_imgs = mminfer.model_single_infer(image,0.9)
    cv2.imshow('Image with Boxes', res_imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()