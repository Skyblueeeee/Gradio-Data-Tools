import os
import re
import time
import datetime
import sqlite3
import numpy as np
from pytz import timezone
from shutil import copy

import fiftyone as fo
import fiftyone.core.odm as foo
import fiftyone.core.utils as fou
from fiftyone import DatasetView
from fiftyone import ViewField as F
foud = fou.lazy_import("fiftyone.utils.data")

from .extends.ann_utils import ViaFile, CocoFile, convert_format

def _format_sep(sentence):
    n_total = 100
    n = round((n_total - len(sentence.encode("gbk")))/2)
    return '-'*n + sentence + '-'*n

def flow_api(CF_fun):
    def wrapper(self, *args, **kwargs):
        tick = time.time()
        try:
            result = CF_fun(self, *args, **kwargs)
        except Exception as e:
            tock = time.time()
            print("Fiftyone Running " + CF_fun.__name__ + f" Finished Time {(tock - tick):.2f}s", end="\n\n")
            raise e
        else:
            tock = time.time()
            print("Fiftyone Running " + CF_fun.__name__ + f" Finished Time {(tock - tick):.2f}s", end="\n\n")
            return result
    return wrapper

class ConstructFlow(object):
    def __init__(self, dataset_name, back_up=True, *args, **kwargs):
        self.dataset_name = dataset_name
        self.back_up = back_up

        # self.ds是很多flow_api运行的基础, 一般从init_dataset方法中返回
        try:
            self.ds = self.init_dataset(self.dataset_name, self.back_up, only_load=True)
        except:
            self.ds = None

    @flow_api
    def init_dataset(self, name, back_up=True, only_load=False):
        """
        备份老的数据库记录, 初始化一个空的数据库记录
        """
        # 只打开
        if only_load:
            return fo.load_dataset(name)

        # 备份
        if back_up:
            try:
                ds = fo.load_dataset(name)
                now_time = str(datetime.datetime.now(timezone('Asia/Shanghai'))).replace(" ", "-").replace(":", "-")[:19]
                backup_name = name + "_bak_" + now_time
                ds.clone(backup_name, True)
            except Exception as e:
                print(f"备份数据库出现异常: {e}")

        ds = None
        # 加载
        try:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)
                ds = fo.Dataset(name)
            else:
                ds = fo.Dataset(name)
            ds.persistent = True
        except Exception as e:
            print(f"加载数据库异常: {e}")

        if ds == None:
            print("数据库加载失败, 程序自动退出! ")
            exit(1)
        return ds

    def search_datasets(self):
        import gradio as gr
        if len(fo.list_datasets()) == 0:
            return gr.Dropdown.update(choices=[""],value="")
        else:
            return gr.Dropdown.update(choices=fo.list_datasets())

    @flow_api
    def delete_datasets(self, dataset_name, mode="equal"):
        """
        删除数据集

        mode是equal模式, 只删除等于dataset_name的数据集
        mode是in模式, 删除名称包含dataset_name的数据集
        """
        if mode == "equal":
            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)
                print(f"已删除dataset--{dataset_name}")
            else:
                print(f"未找到dataset--{dataset_name}")

        elif mode == "in":
            n_del = 0
            for dataset_name_exist in fo.list_datasets():
                if dataset_name in dataset_name_exist:
                    fo.delete_dataset(dataset_name_exist)
                    n_del += 1
                    print(f"已删除dataset--{dataset_name_exist}")
            if n_del == 0:
                print(f"没有dataset名称包含--{dataset_name}")

        elif mode == "all":
            n_del = 0
            for dataset_name_exist in fo.list_datasets():
                fo.delete_dataset(dataset_name_exist)
                n_del += 1
                print(f"已删除dataset--{dataset_name_exist}")
        else:
            raise ValueError(f"mode只支持equa、in、all, 但是收到参数{mode}")

    @flow_api
    def merge(
        self, user_importer, label_field="ground_truth", tags=None, key_field="filepath",
        key_fcn=None, skip_existing=False, insert_new=True, fields=None, omit_fields=None,
        merge_lists=False, overwrite=True, expand_schema=True, add_info=True
    ):
        self.ds.merge_importer(
            user_importer, label_field=label_field, tags=tags, key_field=key_field, key_fcn=key_fcn,
            skip_existing=skip_existing, insert_new=insert_new, fields=fields, omit_fields=omit_fields,
            merge_lists=merge_lists, overwrite=overwrite, expand_schema=expand_schema, add_info=add_info
        )

    @flow_api
    def merge_labels(
        self, dataset_type,
        data_path, labels_path, dataset_dir="", label_field="ground_truth", label_types=["detections"],
        tags=["train"],
        key_field="filename", key_fcn=None,
        skip_existing=False, insert_new=False, fields=None, omit_fields=None, merge_lists=False,
        overwrite=True, expand_schema=True, add_info=True,
        **kwargs
    ):
        from fiftyone.utils.data.importers import _build_parse_sample_fcn, _generate_group_samples, GroupDatasetImporter, parse_dataset_info, LegacyFiftyOneDatasetImporter
        from fiftyone.core.sample import Sample

        dataset_importer, _ = foud.build_dataset_importer(
            dataset_type,
            dataset_dir=dataset_dir,
            data_path=data_path,
            labels_path=labels_path,
            label_types=label_types,
            name=self.ds.name,
            **kwargs,
        )

        with dataset_importer:
            parse_sample, expand_schema, _ = _build_parse_sample_fcn(
                self.ds, dataset_importer, label_field, tags, expand_schema, False
            )
            # 此函数实际都是fiftyone源码, 之所以抽出来, 就是为了换解析函数
            if isinstance(label_field, dict):
                label_key = lambda k: label_field.get(k, k)
            elif label_field is not None:
                label_key = lambda k: label_field + "_" + k
            else:
                label_field = "ground_truth"
                label_key = lambda k: k

            def diy_parse_sample(sample):
                image_path, image_metadata, label = sample
                sample = Sample(
                    filepath=image_path,
                    metadata=image_metadata,
                    tags=tags,
                )
                sample["filename"] = os.path.basename(image_path)

                if isinstance(label, dict):
                    sample.update_fields(
                        {label_key(k): v for k, v in label.items()}
                    )
                elif label is not None:
                    sample[label_field] = label

                return sample

            # 使用自定义的解析函数
            parse_sample = diy_parse_sample

            try:
                num_samples = len(dataset_importer)
            except:
                num_samples = None

            if isinstance(dataset_importer, GroupDatasetImporter):
                samples = _generate_group_samples(dataset_importer, parse_sample)
            else:
                samples = map(parse_sample, iter(dataset_importer))

            self.ds.merge_samples(
                samples,
                key_field=key_field,
                key_fcn=key_fcn,
                skip_existing=skip_existing,
                insert_new=insert_new,
                fields=fields,
                omit_fields=omit_fields,
                merge_lists=merge_lists,
                overwrite=overwrite,
                expand_schema=expand_schema,
                num_samples=num_samples,
            )

            if add_info and dataset_importer.has_dataset_info:
                info = dataset_importer.get_dataset_info()
                if info:
                    parse_dataset_info(self.ds, info)

            if isinstance(dataset_importer, LegacyFiftyOneDatasetImporter):
                dataset_importer.import_run_results(self.ds)

    def run(self):
        # 数据库初始化

        # 原图数据入库

        # filename去重

        # 发图信息入库

        # 标注信息入库

        # 预测信息入库

        # 更新sql库, 以使用sql功能

        # 库信息纠正

        # 导出最终版sql库, 供正常使用
        raise NotImplementedError("将操作流构建在此方法中")

    @staticmethod
    def _get_datasets_id(mongo_db_name, conn):
        dataset_doc = conn.datasets.find_one({"name": mongo_db_name}, {"_id": 1})
        if dataset_doc is None:
            raise ValueError("Dataset '%s' not found" % mongo_db_name)
        return dataset_doc.get("_id", None).__str__()

    @flow_api
    def to_unique(self, key_field):
        """
        将某个field中的值变成唯一的, 只支持字符串类型
        """
        s_dict = {}
        for s in self.ds.select_fields(key_field).iter_samples(progress=True, autosave=True, batch_size=1.0):
            value = s[key_field]
            num = s_dict.get(value, 0)
            if num:
                s[key_field] = value + f"重复{num}"
                s_dict[value] = num + 1
            else:
                s_dict[value] = 1

    @flow_api
    def to_sqlite3(self, sqlite3_db_path,tosqldir):
        def db_connection(sqlite3_db_path):
            try:
                con = sqlite3.connect(sqlite3_db_path)
                return con
            except Exception as e:
                print(f"数据库连接失败: {str(e)}")
        
        def create_table(con, cursor):
            cursor.execute("CREATE TABLE imageInfo (id text PRIMARY KEY)")
            con.commit()

        def add_columns(con, cursor, cols: list, type_dict: dict):
            for each_col in cols:
                cursor.execute("ALTER TABLE imageInfo ADD COLUMN {} {}".format(each_col, type_dict.get(each_col, "text")))
            con.commit()

        def insert_one(con, cursor, infos: dict):
            col_names = "id, "
            values = '"{}", '.format(infos["_id"].__str__())

            col_names_list = []
            values_list = []
            type_dict = {}
            for key in infos:
                if key[0] == "_":
                    continue
                value = infos[key]
                if isinstance(value, str):
                    type_dict[key] = "text"
                elif isinstance(value, int):
                    type_dict[key] = "integer"
                elif isinstance(value, float):
                    type_dict[key] = "real"
                else:
                    continue
                col_names_list.append(key)
                values_list.append(value)
    
            col_names += ", ".join(map(lambda each: '"{}"'.format(each), col_names_list))
            values += ", ".join(map(lambda each: '"{}"'.format(each), values_list))
            sql_sentence = f"INSERT INTO imageInfo ({col_names}) VALUES ({values})"

            loop_flag = True
            while True:
                try:
                    cursor.execute(sql_sentence)
                    loop_flag = False
                except sqlite3.OperationalError as e:
                    m = re.search("table imageInfo has no column named (.+)", str(e))
                    if m:
                        add_columns(con, cursor, [m.group(1)], type_dict)
                    else:
                        print(f"插入数据时发生错误: {str(e)}")
                        exit(1)
                except Exception as e:
                    print(f"插入数据时发生错误: {str(e)}")
                    exit(1)
                if not loop_flag:
                    break

        mongo_conn = foo.get_db_conn()
        collection_real_name = "samples." + self._get_datasets_id(self.dataset_name, mongo_conn)
        if not os.path.exists(sqlite3_db_path):
            os.mkdir(sqlite3_db_path)
        sqlite3_db_path = os.path.join(sqlite3_db_path,"fiftyone.db")
        sqlite3_db_path = os.path.realpath(sqlite3_db_path)
        if os.path.exists(sqlite3_db_path):
            # r = input(f"sqlite3数据库<{sqlite3_db_path}>已存在, 是否删除继续(y/n): ")
            # if r != "y":
            #     exit(0)
            # else:
                os.remove(sqlite3_db_path)

        sqlite3_conn = db_connection(sqlite3_db_path)
        if sqlite3_conn is None:
            print(f"{os.path.dirname(sqlite3_db_path)}文件夹不存在，请手动创建！")
            exit(1)

        sqlite3_cursor = sqlite3_conn.cursor()
        create_table(sqlite3_conn, sqlite3_cursor)

        print("\n正在转换sqlite3")
        pb = fou.ProgressBar(iters_str="iters")
        for b in pb(mongo_conn.get_collection(collection_real_name).find({}).batch_size(200).allow_disk_use(True)):
            insert_one(sqlite3_conn, sqlite3_cursor, b)
        sqlite3_conn.commit()
        print("sqlite3转换完毕\n")
        if tosqldir != "":
            if os.path.exists(tosqldir):
                tosqldirpath = os.path.join(tosqldir,"fiftyone.db")
                import shutil
                shutil.copy(sqlite3_db_path,tosqldirpath)

    @flow_api
    def to_annotation(self, dataset: DatasetView,
        save_dir: str, file_name: str="ff_export.json",
        classes: dict=None,  mode="via", with_picture=True,
        with_label=False, field="ground_truth",
        overwrite=False, skip_failures=False, num_workers=8):
        """
        提取带标注或者不带标注的文件
        mode可以为via或coco, 或它们相加
        """

        # 计算metadata
        if with_label:
            _samples = dataset.select_fields(["filepath", field])
        else:
            _samples = dataset.select_fields(["filepath"])

        _samples.compute_metadata(overwrite=overwrite, skip_failures=skip_failures, num_workers=num_workers)

        # 获取class
        if classes == None:
            try:
                print(f"发现classes为None, 正在从{field}抽取classes,")
                print(f"warning: 不建议使用这种方式, 可能会和预想产生差异!")
                classes = list(set(_samples._dataset.select_fields([field]).distinct(F(f"{field}.detections.label"))))
            except Exception as e:
                print(f"从{field}中抽取classes失败: {e}, 请检查或手动赋值")
                return

        # 都使用via模式, 其他模式再转换标签文件
        via_file = ViaFile.from_init(classes=classes)
        if os.path.exists(save_dir):
            print("路径已存在，请检查！")
        else:
            os.makedirs(save_dir, exist_ok=False)
            for sample in _samples.iter_samples(progress=True):
                filepath, img_size, img_w, img_h = sample["filepath"], sample.metadata["size_bytes"], sample.metadata["width"], sample.metadata["height"]
                if with_label and sample[field] is not None:
                    np_bboxes = np.array([d["bounding_box"] for d in sample[field]["detections"]], dtype=np.float64)
                    np_bboxes[:, [0, 2]] *= img_w
                    np_bboxes[:, [1, 3]] *= img_h
                    bboxes = np_bboxes.tolist()

                    labels = [d["label"] for d in sample[field]["detections"]]

                    scores = [d["confidence"] for d in sample[field]["detections"]]
                    n_none = scores.count(None)
                    if scores.count(None) == len(scores):
                        scores = None
                    elif n_none != 0:
                        raise AssertionError("同一张图的标签score字段不统一!")
                else:
                    bboxes = []
                    labels = []
                    scores = None

                via_file.update_one(
                    img_name=os.path.basename(filepath),
                    img_size=img_size,
                    img_wh=(img_w, img_h),
                    bboxes=bboxes,
                    labels=labels,
                    scores=scores
                )
                if with_picture:
                    copy(filepath, os.path.join(save_dir, os.path.basename(filepath)))
            
            for m in mode.split("+"):
                if m == "via":
                    via_file.save(os.path.join(save_dir, m + "_" + file_name))
                elif m == "coco":
                    coco_file = CocoFile.from_init(classes=via_file.class_names)
                    convert_format(via_file, coco_file)
                    coco_file.save(os.path.join(save_dir, m + "_" + file_name))
                else:
                    print(f"目前只支持via与coco, 发现无法解析的格式: {m}")
                    continue