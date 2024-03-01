"""
统一的标注格式类库:

接口形式统一: update_one以一张图片为主
to_convert: 转换成update_one相同的入口
都有from_init和from_file至少两种实例化方式
含有等量的基础元信息(可以互相转换的基础)
"""

import json

def convert_format(SrcFile, TarFile):
    for per_image_info in SrcFile.to_convert():
        TarFile.update_one(*per_image_info)

def coco_json_init():
    coco_json = {
        "info": {
            "year": 2088,
            "version": "1",
            "description": "Exported from python",
            "contributor": "The World",
            "url": "LoveAndPeace@github",
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
    return coco_json

class CocoFile():
    @classmethod
    def from_init(cls, classes=None, super_category="default"):
        self = cls()

        self.super_category = super_category
        self.coco_json = coco_json_init()

        self.coco_images = self.coco_json["images"]
        self.image_id = len(self.coco_images) + 1
        self.coco_annotations = self.coco_json["annotations"]
        self.ann_id = len(self.coco_annotations) + 1
        self.coco_attrs = self.coco_json["categories"]
        self.update_classes(classes, mode="reset")

        return self

    @classmethod
    def from_file(cls, coco_path):
        self = cls()

        with open(coco_path, "r", encoding="utf-8") as fp:
            self.coco_json = json.load(fp)

        self.coco_images = self.coco_json["images"]
        self.image_id = len(self.coco_images) + 1
        self.coco_annotations = self.coco_json["annotations"]
        self.ann_id = len(self.coco_annotations) + 1
        self.coco_attrs = self.coco_json["categories"]
        if len(self.coco_attrs) == 0:
            self.super_category = "default"
        else:
            self.super_category = self.coco_attrs[0]["supercategory"]

        return self

    def update_classes(self, classes, mode="reset"):
        if mode not in ("reset", "update"):
            print("mode只能是reset或update!")
            return

        if not isinstance(classes, (list, tuple, type(None))):
            print("classes只能是list或tuple或None")
            return
        if classes is None:
            classes = []

        temp_attrs = {l["name"]: l["id"]  for l in self.coco_attrs}
        if mode == "reset":
            temp_attrs.clear()
            category_id = 1
        elif mode == "update":
            category_id = max(temp_attrs.values()) + 1

        for c in classes:
            if c not in temp_attrs:
                temp_attrs[c] = category_id
                category_id += 1
        
        self.coco_attrs.clear()
        temp_attrs = [
            {
                "id": i,
                "name": name,
                "supercategory": self.super_category
            }
            for name, i in temp_attrs.items()
        ]
        self.coco_attrs.extend(sorted(temp_attrs, key=lambda x: x["id"]))

    @property
    def class_names(self):
        return [d["name"] for d in sorted(self.coco_attrs, key=lambda x: x["id"])]

    def update_one(self, img_name, img_size, img_wh, bboxes: list=[], labels: list=[], scores: list=None):
        """
        bboxes: 接受xywh形式, 图片绝对大小
        """
        self.coco_images.append(
            {
                "id": self.image_id,
                "size": img_size,
                "width": img_wh[0],
                "height": img_wh[1],
                "file_name": img_name,
            }
        )

        temp_attrs = {l["name"]: l["id"]  for l in self.coco_attrs}
        for i, (x, y, w, h) in enumerate(bboxes):
            one_ann = {
                "id": self.ann_id,
                "image_id": self.image_id,
                "category_id": temp_attrs[labels[i]],
                "segmentation": [[
                    round(x),
                    round(y),
                    round(x + w),
                    round(y),
                    round(x + w),
                    round(y + h),
                    round(x),
                    round(y + h)
                ]],
                "area": round(w*h),
                "bbox": [round(x), round(y), round(w), round(h)],
                "iscrowd": 0
            }
            if scores != None:
                one_ann["score"] = round(scores[i], 5)
            self.coco_annotations.append(one_ann)
            self.ann_id += 1

        self.image_id += 1

    def save(self, save_path):
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(self.coco_json, fp, ensure_ascii=False)

    def to_convert(self):
        classes_dict = {l["id"]: l["name"]  for l in self.coco_attrs}

        image_ann_dict = {}
        for each_ann in self.coco_annotations:
            temp_list = image_ann_dict.get(each_ann["image_id"], [[], [], []])
            temp_list[0].append(each_ann["bbox"])
            temp_list[1].append(classes_dict[each_ann["category_id"]])
            if each_ann.get("score", None) is not None:
                temp_list[2].append(each_ann["score"])
            image_ann_dict[each_ann["image_id"]] = temp_list
        for k in image_ann_dict:
            l_score = len(image_ann_dict[k][2])
            if l_score == 0:
                image_ann_dict[k][2] = None
            elif l_score != len(image_ann_dict[k][1]):
                raise AssertionError("同一张图的标签score字段不统一!")

        for each_image in self.coco_images:
            try:
                bbox_infos = image_ann_dict[each_image["id"]]
            except KeyError:
                bbox_infos = [[], [], None]

            yield (
                each_image["file_name"],
                each_image.get("size", None),
                (each_image["width"], each_image["height"]),
                *bbox_infos
            )

def via_json_init(super_category):
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
                super_category: {
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
    return via_json

class ViaFile():
    @classmethod
    def from_init(cls, classes=None, super_category="default"):
        self = cls()

        self.super_category = super_category
        self.via_json = via_json_init(super_category)

        # 初始化标签信息
        self.via_imgs = self.via_json["_via_img_metadata"]
        self.via_attrs = self.via_json["_via_attributes"]["region"][super_category]["options"]
        self.update_classes(classes, mode="reset")

        return self

    @classmethod
    def from_file(cls, via_path):
        self = cls()
        with open(via_path, "r", encoding="utf-8") as fp:
            self.via_json = json.load(fp)
        self.super_category = list(self.via_json["_via_attributes"]["region"].keys())[0]
        self.via_attrs = self.via_json["_via_attributes"]["region"][self.super_category]["options"]
        self.via_imgs = self.via_json["_via_img_metadata"]
        return self

    def update_one(self, img_name, img_size, img_wh, bboxes: list=[], labels: list=[], scores: list=None):
        """
        bboxes: 接受xywh形式, 图片绝对大小
        """
        self.via_imgs[img_name + str(img_size)] = {
            "filename": img_name,
            "size": img_size,
            "width": img_wh[0],
            "height": img_wh[1],
            "regions": [],
            "file_attributes": {}
        }

        regions = self.via_imgs[img_name + str(img_size)]["regions"]
        for i, (x, y, w, h) in enumerate(bboxes):
            regions.append(
                {
                    "shape_attributes": {
                        "name": "rect",
                        "x": round(x),
                        "y": round(y),
                        "width": round(w),
                        "height": round(h)
                    },
                    "region_attributes": {
                        self.super_category: labels[i]
                    }
                }
            )
            if scores != None:
                regions[i]["shape_attributes"]["score"] = round(scores[i], 5)

    def save(self, save_path):
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(self.via_json, fp, ensure_ascii=False)

    def update_classes(self, classes, mode="reset"):
        if mode not in ("reset", "update"):
            print("mode只能是reset或update!")
            return

        if mode == "reset":
            self.via_attrs.clear()
        elif mode == "update":
            pass

        if isinstance(classes, dict):
            for k, v in classes.items():
                self.via_attrs[k] = v
        elif classes != None:
            for k in classes:
                self.via_attrs[k] = ""
        else:
            pass
    
    def to_convert(self):
        for img_infos in self.via_imgs.values():
            scores = [r["shape_attributes"].get("score", None) for r in img_infos["regions"]]
            n_none = scores.count(None)
            if scores.count(None) == len(scores):
                scores = None
            elif n_none != 0:
                raise AssertionError("同一张图的标签score字段不统一!")

            yield (
                img_infos["filename"],
                img_infos["size"],
                (img_infos.get("width", None), img_infos.get("height", None)),
                [[r["shape_attributes"]["x"], r["shape_attributes"]["y"], r["shape_attributes"]["width"], r["shape_attributes"]["height"]] for r in img_infos["regions"]],
                [r["region_attributes"][self.super_category] for r in img_infos["regions"]],
                scores
            ) 

    @property
    def class_names(self):
        return list(self.via_attrs.keys())
