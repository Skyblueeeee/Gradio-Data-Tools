import os
import fiftyone as fo
import fiftyone.utils.data as foud
import fiftyone.core.labels as fol
from fiftyone.types.dataset_types import ImageDetectionDataset
from .extends.ann_utils import ViaFile
import numpy as np
import cv2

class FitowBaseImporter(foud.GenericSampleDatasetImporter):
    def __init__(self, *args, **kwargs):
        pass
    
    def _parse_one(self):
        raise NotImplementedError("需要写一个解析生成器(核心代码)")

    def __iter__(self):
        self._rel_iter = self._parse_one()
        return self
    
    def __next__(self):
        # img_path, img_name主要用来去重，之所以还加一个filename，因为其可能是编码的，与basename并不一定一样
        img_path, img_name, img_info_dict = next(self._rel_iter)
        return fo.Sample(filepath=img_path, filename=img_name, **img_info_dict)

    def has_dataset_info(self):
        return False

    def get_dataset_info(self):
        return None

    def has_sample_field_schema(self):
        return True

    def get_sample_field_schema(self):
        field_schema = {
            "filename": "fiftyone.core.fields.StringField",
        }
        raise NotImplementedError("自己填新增字段")
        return field_schema

class FitowBaseLabelImporter(foud.LabeledImageDatasetImporter):
    def __init__(self, data_path, labels_path, label_types="", dataset_dir=None, shuffle=False, seed=None, max_samples=None):
        super().__init__(dataset_dir, shuffle, seed, max_samples)
        self.data_path = data_path
        self.labels_path = labels_path

    def _parse_one_image(self):
        raise NotImplementedError("需要写一个解析生成器(核心代码)")

    @property
    def label_cls(self):
        label_field_schema = {
            # "filepath": "fiftyone.core.fields.StringField",
            "detections": "fiftyone.core.labels.Detections",
        }
        raise NotImplementedError("需要指明字段类型")
        return label_field_schema

    def __len__(self):
        return NotImplementedError("需要解析长度")

    def __iter__(self):
        self._rel_iter = self._parse_one_image()
        return self
    
    def __next__(self):
        img_path, img_label_info_dict = next(self._rel_iter)
        if len(img_label_info_dict) == 1 and "detections" in img_label_info_dict:
            img_label_info_dict = img_label_info_dict["detections"]
        return img_path, None, img_label_info_dict

    @property
    def has_dataset_info(self):
        return False

    def get_dataset_info(self):
        return None

    @property
    def has_image_metadata(self):
        return False

class ViaLabelImporter(FitowBaseLabelImporter):
    def __init__(self, data_path, labels_path, label_types="", dataset_dir=None, shuffle=False, seed=None, max_samples=None):
        super().__init__(data_path, labels_path, label_types, dataset_dir, shuffle, seed, max_samples)

        real_labels_path = None
        if os.path.exists(labels_path):
            real_labels_path = labels_path
        elif os.path.exists(os.path.join(data_path, labels_path)):
            real_labels_path = os.path.join(data_path, labels_path)
        else:
            raise FileExistsError("标签文件路径不存在!")

        self.via_file = ViaFile.from_file(real_labels_path)

    def _parse_one_image(self):
        for each_img_info in self.via_file.via_imgs.values():
            if "width" not in each_img_info:
                image = cv2.imdecode(np.fromfile(os.path.join(self.data_path, each_img_info['filename']), dtype=np.uint8),-1)
                img_h,img_w= image.shape
            else:
                img_w, img_h = each_img_info["width"], each_img_info["height"]
            regions = each_img_info["regions"]
            detections = []
            for region in regions:
                label = region["region_attributes"][self.via_file.super_category]
                region = region["shape_attributes"]
                region.pop("name", None)
                x, y, w, h = region.pop("x"), region.pop("y"), region.pop("width"), region.pop("height")
                bounding_box = [x / img_w, y / img_h, w / img_w, h / img_h]
                detections.append(fol.Detection(
                    label=label,
                    bounding_box=bounding_box,
                    confidence=region.pop("score", None),
                    **region
                ))
            detections = fol.Detections(detections=detections)
            yield os.path.join(self.data_path, each_img_info["filename"]), {"detections": detections}

    def __len__(self):
        return len(self.via_file.via_imgs)

    @property
    def label_cls(self):
        label_field_schema = {
            "detections": "fiftyone.core.labels.Detections",
        }
        return label_field_schema
        
class ViaLabelTypes(ImageDetectionDataset):
    def get_dataset_importer_cls(self):
        return ViaLabelImporter

    def get_dataset_exporter_cls(self):
        return None
    
class OldViaLabelImporter(FitowBaseLabelImporter):
    def __init__(self, data_path, labels_path, label_types="", dataset_dir=None, shuffle=False, seed=None, max_samples=None):
        super().__init__(data_path, labels_path, label_types, dataset_dir, shuffle, seed, max_samples)

        real_labels_path = None
        if os.path.exists(labels_path):
            real_labels_path = labels_path
        elif os.path.exists(os.path.join(data_path, labels_path)):
            real_labels_path = os.path.join(data_path, labels_path)
        else:
            raise FileExistsError("标签文件路径不存在!")

        self.via_file = ViaFile.from_file(real_labels_path)

    def _parse_one_image(self):
        for each_img_info in self.via_file.via_imgs.values():

            image = cv2.imdecode(np.fromfile(os.path.join(self.data_path, each_img_info['filename']), dtype=np.uint8),-1)
            img_h,img_w, _ = image.shape

            regions = each_img_info["regions"]
            detections = []
            for region in regions:
                label = region["region_attributes"][self.via_file.super_category]
                region = region["shape_attributes"]
                region.pop("name", None)
                x, y, w, h = region.pop("x"), region.pop("y"), region.pop("width"), region.pop("height")
                bounding_box = [x / img_w, y / img_h, w / img_w, h / img_h]
                detections.append(fol.Detection(
                    label=label,
                    bounding_box=bounding_box,
                    confidence=region.pop("score", None),
                    **region
                ))
            detections = fol.Detections(detections=detections)
            yield os.path.join(self.data_path, each_img_info["filename"]), {"detections": detections}

    def __len__(self):
        return len(self.via_file.via_imgs)

    @property
    def label_cls(self):
        label_field_schema = {
            "detections": "fiftyone.core.labels.Detections",
        }
        return label_field_schema

class OldViaLabelTypes(ImageDetectionDataset):
    def get_dataset_importer_cls(self):
        return OldViaLabelImporter

    def get_dataset_exporter_cls(self):
        return None
    