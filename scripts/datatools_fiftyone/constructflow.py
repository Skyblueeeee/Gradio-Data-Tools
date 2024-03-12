import os
# 环境变量设置配置文件地址
os.environ["FIFTYONE_CONFIG_PATH"] = "scripts/datatools_fiftyone/fo_config.json"

# scripts.fiftyone_win

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.types import COCODetectionDataset
import scripts.datatools_fiftyone.fiftyone_dir as ff
# import fiftyone_dir as ff

import json
import os,sys
import gradio as gr

# fiftyone数据库发图代码
def db_tags(db_name):
    from datetime import datetime
    current_time = datetime.now()
    sendtime = current_time.strftime("发图时间_%y%m%d")
    if db_name == "":
        return "",sendtime,"数据库名称未填写"

    cf = GRConstuctFlow(db_name, back_up=False)
    tags = list(cf.ds.count_sample_tags().keys())
    
    return gr.Dropdown.update(choices=tags,value=tags[0]),sendtime

def send_image(db_name,tag,send_dir,send_date,via_classes):
    if db_name == "":
        return "数据库名称未填写"
    if tag == "" or send_date == "":
        return "发图信息未填写完整"
    if not os.path.exists(send_dir):
        return "请检查发图地址！未有该路径。"
    if os.path.exists(os.path.join(send_dir,send_date)):
        return "请检查发图路径！已有发图数据。"
    label_dict = {}
    for group in via_classes.to_dict("split")["data"]:
        if group[0]!= "":
            label_dict[group[0]]=group[1]

    cf = GRConstuctFlow(db_name, back_up=False)
    ds_view = cf.ds.match_tags([tag])
    one_view = ds_view.match(F("Tag")=="标注图")
    if len(one_view) != 0:
        via_path = os.path.join(send_dir,send_date)
        cf.to_annotation(
            one_view,
            via_path,
            mode="via",
            overwrite=False,
            skip_failures=False,
            num_workers=16,
            classes=label_dict,
            with_label=False,
            with_picture=True
        )

    return f"发图任务已完成! 该次抽取的图片数量为{len(one_view)}"

# 构建user importer
class GROriginDatasetImporter(ff.FitowBaseImporter):
    def __init__(self, dataset_dir,rule=""):
        self.dataset_dir = dataset_dir
        self.rule = rule

    def _parse_one(self):
        N_dir = 0
        for group in os.walk(self.dataset_dir):
            if self.rule in group[0]:
                for file in os.listdir(group[0]):
                    if file.endswith((".jpg",".bmp",".png")):
                        json_info = {"Origin_GR":"原图"}

                        yield os.path.join(group[0], file),file, json_info

    def get_sample_field_schema(self):
        field_schema = {
            "filename": "fiftyone.core.fields.StringField",
            "Origin_GR":"fiftyone.core.fields.StringField",
        }
        return field_schema

class GRFileAttrVIADatasetImporter(ff.FitowBaseImporter):
    def __init__(self, via_path,rule="", *args, **kwargs):
        self.via_path = via_path

    def _parse_one(self):
        print(f"\n正在读取project文件信息: {self.via_path}")
        with open(self.via_path, "r", encoding="utf-8") as fp:
            via_dict = json.load(fp)

            img_dict = via_dict["_via_img_metadata"]
            for each_img_info in img_dict.values():
                each_img = each_img_info["filename"]
                each_file_attr = each_img_info["file_attributes"].get("fileattr", "有效")

                if len(each_img_info["regions"]) == 0:
                    yield each_img, each_img, {"VIAFileAttr": "空图"}
                else:
                    yield each_img, each_img, {"VIAFileAttr": each_file_attr}

    def get_sample_field_schema(self):
        field_schema = {
            "filename": "fiftyone.core.fields.StringField",
            "VIAFileAttr": "fiftyone.core.fields.StringField",
        }
        return field_schema

class GRFileAttrCOCODatasetImporter(ff.FitowBaseImporter):
    def __init__(self, coco_path,rule="", *args, **kwargs):
        self.coco_path = coco_path

    def _parse_one(self):
        print(f"\n正在读取project文件信息: {self.coco_path}")
        with open(self.coco_path, 'r',encoding="utf-8") as file:
            json_data = json.load(file)
            coco_images = json_data["images"]
            coco_ann = json_data["annotations"]

        ann_img_id_list =[]
        for ann in coco_ann:
            ann_img_id_list.append(int(ann["image_id"]))

        for image in coco_images:
            if image["id"] in ann_img_id_list:
                yield image["file_name"], image["file_name"], {"COCOFileAttr": "有效"}
            else:
                yield image["file_name"], image["file_name"], {"COCOFileAttr": "空图"}

    def get_sample_field_schema(self):
        field_schema = {
            "filename": "fiftyone.core.fields.StringField",
            "COCOFileAttr": "fiftyone.core.fields.StringField",
        }
        return field_schema

# 构建construct flow
class GRConstuctFlow(ff.ConstructFlow):
    def __init__(self, dataset_name, back_up=True, *args, **kwargs):
        super().__init__(dataset_name, back_up, *args, **kwargs)

    def run(self, origin_dir="",imgdir_rule="",file_uniqe = 0,init_dateset = 0,merge_origin = 0,merge_label=0,label_attr=0,via_coco_mode="COCO",to_sql = 0,to_sql_dir="",del_db=0,del_mode="equal"):
        # 数据库初始化
        if init_dateset:
            self.ds = self.init_dataset(self.dataset_name, self.back_up)

        # 图片录入
        if merge_origin:
            self.merge(
                GROriginDatasetImporter(
                    dataset_dir=origin_dir,rule=imgdir_rule),tags=["origin"])
        
        # 图片去重
        if file_uniqe:
            self.to_unique("filename")

        # 标注有效性
        if label_attr:
            for group in os.walk(origin_dir):
                if imgdir_rule in group[0]:
                    for each_file in os.listdir(group[0]):
                        if "via" in each_file and via_coco_mode == "VIA":
                            via_path = os.path.join(group[0],each_file)
                            print(f"Fiftyone Runing {via_path} Valid Images")
                            self.merge(
                                GRFileAttrVIADatasetImporter(via_path=via_path),
                                key_field="filename",
                                omit_fields=["filepath"],
                                insert_new=True
                            )
                        if "coco" in each_file and via_coco_mode == "COCO":
                            coco_path = os.path.join(group[0],each_file)
                            print(f"Fiftyone Runing {coco_path} Valid Images")
                            self.merge(
                                GRFileAttrCOCODatasetImporter(coco_path=coco_path),
                                key_field="filename",
                                omit_fields=["filepath"],
                                insert_new=True
                            )

        # coco格式标注信息入库
        if merge_label and via_coco_mode == "COCO":
            import fiftyone as fo
            for group in os.walk(origin_dir):
                if imgdir_rule in group[0]:
                    for each_file in os.listdir(group[0]):
                        if "coco" in each_file and each_file.endswith(".json"):
                            coco_path = os.path.join(group[0],each_file)
                            print(f"Fiftyone Runing {coco_path} labels")
                            self.merge_labels(
                                dataset_type=fo.types.COCODetectionDataset,
                                data_path=group[0],
                                labels_path= coco_path,
                                label_field="ground_truth",
                                label_types=["detections"],
                                tags=["train"],
                                key_field="filename",
                                key_fcn=None,
                                skip_existing=False,
                                insert_new=False,
                                fields=None,
                                omit_fields=["filepath"],
                                merge_lists=False,
                                overwrite=True,
                                expand_schema=True,
                                add_info=True
                            )

        # via格式标注信息入库(直接使用包含width和height的via导入,没有则读取图片)
        if merge_label and via_coco_mode == "VIA":
            import fiftyone as fo
            for group in os.walk(origin_dir):
                if imgdir_rule in group[0]:
                    for each_file in os.listdir(group[0]):
                        if "via" in each_file and each_file.endswith(".json"):
                            label_path = os.path.join(group[0], each_file)
                            print(f"Fiftyone Runing {label_path} labels")
                            self.merge_labels(
                                dataset_type=ff.construct.ViaLabelTypes,
                                data_path=group[0],
                                labels_path= label_path,
                                label_field="ground_truth",
                                label_types=["detections"],
                                tags=["train"],
                                key_field="filename",
                                key_fcn=None,
                                skip_existing=False,
                                insert_new=False,
                                fields=None,
                                omit_fields=["filepath", "metadata"],
                                merge_lists=False,
                                overwrite=True,
                                expand_schema=True,
                                add_info=True
                            )
        
        # yolo格式标注信息入库
        if merge_label and via_coco_mode == "YOLO":
            import fiftyone as fo
            for group in os.walk(origin_dir):
                if imgdir_rule in group[0]:
                    for each_file in os.listdir(group[0]):
                        if each_file == "data.yaml":
                            label_path = os.path.join(group[0], each_file)
                            print(f"Fiftyone Runing {label_path} labels")
                            self.merge_labels(
                                dataset_type=fo.types.YOLOv5Dataset,
                                dataset_dir=group[0],
                                yaml_path=label_path,
                                split="val",
                                data_path=group[0],
                                labels_path= label_path,
                                label_field="ground_truth",
                                label_types=["detections"],
                                tags=["train"],
                                key_field="filename",
                                key_fcn=None,
                                skip_existing=False,
                                insert_new=False,
                                fields=None,
                                omit_fields=["filepath", "metadata"],
                                merge_lists=False,
                                overwrite=True,
                                expand_schema=True,
                                add_info=True
                            )

        # 导出最终版sql库，供正常使用
        if to_sql:
            import subprocess
            self.to_sqlite3("scripts/fiftyone_win/fo_data/sqlite3",to_sql_dir)
            # scp_command = f"scp /root/sqlite3/fitow_fiftyone.db /root/workspace/fo_data/sqlite3"
            # subprocess.run(scp_command, shell=True, check=True)
        
        # 删除数据库(谨慎使用)
        if del_db:
            self.delete_datasets(dataset_name=self.dataset_name,mode=del_mode)

if __name__ == "__main__":
    db_name = "hanfeng"
    send_dir = "D:/AI"
    send_date ="发图_240112"
    db_backup = False
    ori_dir = r"D:\20220819"
    via_classes = ""

    cf = GRConstuctFlow(db_name, back_up=db_backup)
    cf.run( origin_dir= ori_dir,
            init_dateset = 1,merge_origin = 1,  
            merge_label  = 1,label_attr =0,via_coco_mode="YOLO"
    )
    

    # send_image(db_name,send_dir,send_date,via_classes)