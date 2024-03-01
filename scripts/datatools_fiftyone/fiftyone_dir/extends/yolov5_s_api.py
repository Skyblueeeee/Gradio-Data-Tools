import os
import time
import json
import numpy as np
from threading import Thread
from functools import reduce
from .yolov5_metrics import ConfusionMatrix, ap_per_class, process_batch

class ReadNumpyGroup():
    def __init__(self, save_dir, save_name="numpy_group", dtype=np.float32):
        self.save_dir = save_dir
        self.save_name = save_name

        with open(os.path.join(self.save_dir, self.save_name + ".json"), "r") as fp:
            self._json = json.load(fp)

        self._dtype = dtype
        self._itemsize = np.zeros(0, dtype=dtype).itemsize
        if not self._check():
            raise AssertionError("形状长度与存储长度不匹配, 请检查dtype")

    def _check(self):
        json_size = 0
        for group in self._json:
            for g in group:
                json_size += reduce(lambda x, y: x*y, g)
        json_size = json_size*self._itemsize
        array_size = os.stat(os.path.join(self.save_dir, self.save_name + ".npz")).st_size
        return json_size == array_size

    def read_groups(self):
        with open(os.path.join(self.save_dir, self.save_name + ".npz"), "rb") as fp:
            for group in self._json:
                np_group = []
                for shape in group:
                    size = reduce(lambda x, y: x*y, shape)*self._itemsize
                    nnn = np.frombuffer(fp.read(size), dtype=self._dtype).reshape(shape)
                    np_group.append(nnn)
                yield np_group

class SaveNumpyGroup():
    def __init__(self, save_dir, save_name="numpy_group"):
        self.save_dir = save_dir
        self.save_name = save_name

        self._stop = False
        self._json = []
        self._buffer = []
        self._fp = open(os.path.join(self.save_dir, self.save_name + ".npz"), "wb")

        self._t = Thread(target=self._save_one_group)
        self._t.start()

    def save_one_group(self, *group):
        self._buffer.append(group)

    def close(self):
        self._stop = True
        self._t.join()
        self._fp.close()
        with open(os.path.join(self.save_dir, self.save_name + ".json"), "w") as fp:
            json.dump(self._json, fp)

    def _save_one_group(self):
        while True:
            l = len(self._buffer)
            if self._stop and l == 0:
                break

            if len(self._buffer) != 0:
                group = self._buffer.pop(0)
                for g in group:
                    self._fp.write(g.tobytes())
                self._json.append([g.shape for g in group])
            else:
                time.sleep(0.1)

class Metrics():
    def __init__(self, save_dir, class_names: dict, n_classes=80, single_cls=False) -> None:
        self.single_cls = single_cls
        self.nc = 1 if self.single_cls else n_classes  # number of classes
        self.save_dir = save_dir
        self.names = class_names

        # variable init
        self.seen = 0
        self.stats = []
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)

    def process_batch(self, detections: np.ndarray, labels: np.ndarray):
        """
        detections nx6 float32 xyxycl
        labels nx5 float32 lxyxy
        """
        self.seen += 1
        # Run NMS

        # details
        iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.shape[0]

        nl = labels.shape[0]
        tcls = labels[:, 0].tolist() if nl else []  # target class
        if detections.shape[0] == 0:
            if nl:
                self.stats.append((np.zeros([0, niou], dtype=np.bool_), np.array([]), np.array([]), tcls))
            return

        if self.single_cls:
            detections[:, 5] = 0
        
        if nl:
            correct = process_batch(detections, labels, iouv)
            self.confusion_matrix.process_batch(detections, labels)
        else:
            correct = np.zeros([detections.shape[0], niou], dtype=np.bool_)
        self.stats.append((correct, detections[:, 4], detections[:, 5], tcls))  # (correct, conf, pcls, tcls)

    def output(self):
        p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ap, ap_class = [], []
        ap50 = []

        self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(self.stats) and self.stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*self.stats, plot=True, save_dir=self.save_dir, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(self.stats[3].astype(np.int64), minlength=self.nc)  # number of targets per class
        else:
            nt = np.zeros(1)

        print(self._format_ap_output((nt, mp, mr, map50, ap_class, p, r, ap50, ap, map), to_save=True))
        self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))
        
    def _format_ap_output(self, args, to_save=True):
        nt, mp, mr, map50, ap_class, p, r, ap50, ap, map = args
        if nt.sum() == 0:
            print("WARNING: no labels found in {task} set, can not compute metrics without labels")

        output_str = ""
        pf = "{:>20}{:>11d}{:>11d}{:>15.3f}{:>11.3f}{:>15.3f}{:>15.3f}"
        output_str += f'{"ClassName":>20}{"images":>11}{"labels":>11}{"precision":>15}{"recall":>11}{"AP@0.5":>15}{"AP@0.5:0.95":>15}\n'
        output_str += (pf.format('all', self.seen, nt.sum(), mp, mr, map50, map)+'\n')
        for i, c in enumerate(ap_class):
            output_str += (pf.format(self.names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i])+'\n')

        if to_save:
            with open(os.path.join(self.save_dir,'PR_recall.txt'), "w", encoding="utf-8") as fp:
                fp.write(output_str)
        return output_str