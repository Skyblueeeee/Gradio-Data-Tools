"""
COCO-style detection evaluation.

| Copyright 2017-2022, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
from collections import defaultdict

import numpy as np

import eta.core.utils as etau

import fiftyone.core.plots as fop
import fiftyone.utils.iou as foui

from .detection import (
    DetectionEvaluation,
    DetectionEvaluationConfig,
    DetectionResults,
)


logger = logging.getLogger(__name__)


class COCOFitowEvaluationConfig(DetectionEvaluationConfig):
    """COCO-style evaluation config.

    Args:
        pred_field: the name of the field containing the predicted
            :class:`fiftyone.core.labels.Detections` or
            :class:`fiftyone.core.labels.Polylines`
        gt_field: the name of the field containing the ground truth
            :class:`fiftyone.core.labels.Detections` or
            :class:`fiftyone.core.labels.Polylines`
        iou (None): the IoU threshold to use to determine matches
        classwise (None): whether to only match objects with the same class
            label (True) or allow matches between classes (False)
        iscrowd ("iscrowd"): the name of the crowd attribute
        use_masks (False): whether to compute IoUs using the instances masks in
            the ``mask`` attribute of the provided objects, which must be
            :class:`fiftyone.core.labels.Detection` instances
        use_boxes (False): whether to compute IoUs using the bounding boxes
            of the provided :class:`fiftyone.core.labels.Polyline` instances
            rather than using their actual geometries
        tolerance (None): a tolerance, in pixels, when generating approximate
            polylines for instance masks. Typical values are 1-3 pixels
        compute_mAP (False): whether to perform the necessary computations so
            that mAP and PR curves can be generated
        iou_threshs (None): a list of IoU thresholds to use when computing mAP
            and PR curves. Only applicable when ``compute_mAP`` is True
        max_preds (None): the maximum number of predicted objects to evaluate
            when computing mAP and PR curves. Only applicable when
            ``compute_mAP`` is True
        error_level (1): the error level to use when manipulating instance
            masks or polylines. Valid values are:

            -   0: raise geometric errors that are encountered
            -   1: log warnings if geometric errors are encountered
            -   2: ignore geometric errors

            If ``error_level > 0``, any calculation that raises a geometric
            error will default to an IoU of 0
    """

    def __init__(
        self,
        pred_field,
        gt_field,
        iou=None,
        eval_mode="nv1",
        classwise=None,
        iscrowd="iscrowd",
        use_masks=False,
        use_boxes=False,
        tolerance=None,
        compute_mAP=False,
        iou_threshs=None,
        max_preds=None,
        error_level=1,
        **kwargs,
    ):
        super().__init__(
            pred_field, gt_field, iou=iou, classwise=classwise, **kwargs
        )

        if compute_mAP and iou_threshs is None:
            iou_threshs = [x / 100 for x in range(50, 100, 5)]

        if compute_mAP and max_preds is None:
            max_preds = 100

        self.iscrowd = iscrowd
        self.use_masks = use_masks
        self.use_boxes = use_boxes
        self.tolerance = tolerance
        self.compute_mAP = compute_mAP
        self.iou_threshs = iou_threshs
        self.max_preds = max_preds
        self.error_level = error_level
        self.eval_mode = eval_mode

    @property
    def method(self):
        return "cocofitow"


class COCOFitowEvaluation(DetectionEvaluation):
    """COCO-style evaluation.

    Args:
        config: a :class:`COCOEvaluationConfig`
    """

    def __init__(self, config):
        super().__init__(config)

        if config.iou is None:
            raise ValueError(
                "You must specify an `iou` threshold in order to run COCO "
                "evaluation"
            )

        if config.classwise is None:
            raise ValueError(
                "You must specify a `classwise` value in order to run COCO "
                "evaluation"
            )

    def evaluate(self, sample_or_frame, eval_key=None):
        """Performs COCO-style evaluation on the given image.

        Predicted objects are matched to ground truth objects in descending
        order of confidence, with matches requiring a minimum IoU of
        ``self.config.iou``.

        The ``self.config.classwise`` parameter controls whether to only match
        objects with the same class label (True) or allow matches between
        classes (False).

        If a ground truth object has its ``self.config.iscrowd`` attribute set,
        then the object can have multiple true positive predictions matched to
        it.

        Args:
            sample_or_frame: a :class:`fiftyone.core.sample.Sample` or
                :class:`fiftyone.core.frame.Frame`
            eval_key (None): the evaluation key for this evaluation

        Returns:
            a list of matched
            ``(gt_label, pred_label, iou, pred_confidence, gt_id, pred_id)``
            tuples
        """
        gts = sample_or_frame[self.gt_field]
        preds = sample_or_frame[self.pred_field]

        if eval_key is None:
            # Don't save results on user's data
            eval_key = "eval"
            gts = _copy_labels(gts)
            preds = _copy_labels(preds)

        return _coco_evaluation_single_iou_numpy(gts, preds, eval_key, self.config)

    def generate_results(
        self, samples, matches, eval_key=None, classes=None, missing=None
    ):
        """Generates aggregate evaluation results for the samples.

        If ``self.config.compute_mAP`` is True, this method performs COCO-style
        evaluation as in :meth:`evaluate` to generate precision and recall
        sweeps over the range of IoU thresholds in ``self.config.iou_threshs``.
        In this case, a :class:`COCODetectionResults` instance is returned that
        can compute mAP and PR curves.

        Args:
            samples: a :class:`fiftyone.core.collections.SampleCollection`
            matches: a list of
                ``(gt_label, pred_label, iou, pred_confidence, gt_id, pred_id)``
                matches. Either label can be ``None`` to indicate an unmatched
                object
            eval_key (None): the evaluation key for this evaluation
            classes (None): the list of possible classes. If not provided, the
                observed ground truth/predicted labels are used for results
                purposes
            missing (None): a missing label string. Any unmatched objects are
                given this label for results purposes

        Returns:
            a :class:`DetectionResults`
        """
        config = self.config
        gt_field = config.gt_field
        pred_field = config.pred_field

        # if not config.compute_mAP:
        #     return DetectionResults(
        #         matches,
        #         eval_key=eval_key,
        #         gt_field=gt_field,
        #         pred_field=pred_field,
        #         classes=classes,
        #         missing=missing,
        #         samples=samples,
        #     )

        (
            precision,
            recall,
            thresholds,
            iou_threshs,
            classes,
        ) = _compute_pr_curves_fitow(matches, config, classes=classes)


        _classwise_AP = np.mean(precision, axis=(0, 2))
        _classwise_ap_dict = {}
        for i, c in enumerate(classes):
            _classwise_ap_dict[c] = {
                "precision": -1,
                "recall": -1,
                "AP": float(_classwise_AP[i]),
                "平均IOU": 0.0,
                "平均CONF": 0.0,
                "标签数量": 0,
                "正确检出": 0,
                "漏检数": 0,
                "正确预测": 0,
                "过杀数": 0,
                "误检数": 0,
            }
        ap_dict = _record_for_report(matches, _classwise_ap_dict)

        return COCOFitowDetectionResults(
            matches,
            precision,
            recall,
            iou_threshs,
            classes,
            thresholds=thresholds,
            eval_key=eval_key,
            gt_field=gt_field,
            pred_field=pred_field,
            missing=missing,
            samples=samples,
            ap_dict=ap_dict
        )

# by Joyphy
def _compute_pr_curves_fitow(matches, config, classes=None):
    iou_threshs = config.iou_threshs

    num_threshs = len(iou_threshs)
    thresh_matches = [{} for _ in range(num_threshs)]

    if classes is None:
        _classes = set()

    for idx, matches in enumerate([matches]):
        for match in matches:
            gt_label = match[0]
            pred_label = match[1]

            if classes is None:
                _classes.add(gt_label)
                _classes.add(pred_label)

            if gt_label is None:
                c = pred_label
            elif pred_label is None:
                c = gt_label
            elif gt_label == pred_label:
                c = gt_label
            elif gt_label != pred_label:
                c = pred_label

            if c not in thresh_matches[idx]:
                thresh_matches[idx][c] = {
                    "tp": [],
                    "fp": [],
                    "num_gt": 0,
                }

            if gt_label is None:
                thresh_matches[idx][c]["fp"].append(match)
            elif pred_label is None:
                thresh_matches[idx][c]["num_gt"] += 1
            elif gt_label == pred_label:
                thresh_matches[idx][c]["tp"].append(match)
                thresh_matches[idx][c]["num_gt"] += 1
            elif gt_label != pred_label:
                thresh_matches[idx][c]["fp"].append(match)

    if classes is None:
        _classes.discard(None)
        classes = sorted(_classes)

    num_classes = len(classes)
    class_idx_map = {c: idx for idx, c in enumerate(classes)}

    # Compute precision-recall
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    precision = -np.ones((num_threshs, num_classes, 101))
    thresholds = -np.ones((num_threshs, num_classes, 101))
    recall = np.linspace(0, 1, 101)
    for idx, _thresh_matches in enumerate(thresh_matches):
        for c, matches in _thresh_matches.items():
            c_idx = class_idx_map.get(c, None)
            num_gt = matches["num_gt"]

            if c_idx is None or num_gt == 0:
                continue

            tp = matches["tp"]
            fp = matches["fp"]
            tp_fp = np.array([1] * len(tp) + [0] * len(fp))
            confs = np.array([m[3] for m in tp] + [m[3] for m in fp])
            if None in confs:
                raise ValueError(
                    "All predicted objects must have their `confidence` "
                    "attribute populated in order to compute precision-recall "
                    "curves"
                )

            inds = np.argsort(-confs, kind="mergesort")
            tp_fp = tp_fp[inds]
            confs = confs[inds]

            tp_sum = np.cumsum(tp_fp).astype(dtype=float)
            total = np.arange(1, len(tp_fp) + 1).astype(dtype=float)

            pre = tp_sum / total
            rec = tp_sum / num_gt

            for i in range(len(pre) - 1, 0, -1):
                if pre[i] > pre[i - 1]:
                    pre[i - 1] = pre[i]

            q = np.zeros(101)
            t = np.zeros(101)

            inds = np.searchsorted(rec, recall, side="left")

            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pre[pi]
                    t[ri] = confs[pi]
            except:
                pass

            precision[idx][c_idx] = q
            thresholds[idx][c_idx] = t

    return precision, recall, thresholds, iou_threshs, classes

def _record_for_report(matches, _classwise_ap_dict):
    ap_dict = {}
    if matches:
        # gt_label, dt_label, ious, confs, gt_id, dt_id
        np_matches = np.array(matches, dtype=np.object0)
        none_mask = (np_matches==None)
        np_matches[none_mask[:, 0], 0] = ""
        np_matches[none_mask[:, 1], 1] = ""
        np_matches[none_mask[:, 2], 2] = 0
        np_matches[none_mask[:, 3], 3] = 0
        np_matches[none_mask[:, 4], 4] = ""
        np_matches[none_mask[:, 5], 5] = ""
        classes = set(np.union1d(np_matches[:, 0], np_matches[:, 1]).tolist())
        classes.discard("")
        for c in classes:
            ap_dict[c] = {
                "precision": -1,
                "recall": -1,
                "AP": -1,
                "平均IOU": 0.0,
                "平均CONF": 0.0,
                "标签数量": 0,
                "正确检出": 0,
                "漏检数": 0,
                "正确预测": 0,
                "过杀数": 0,
                "误检数": 0,
            }
        ap_dict.update(_classwise_ap_dict)


        # 得到gt数量
        np_matches = np_matches[np_matches[:, 3].argsort()[::-1]] # 置信度排序
        gt_unique_id = np.unique(np_matches[:, 4], return_index=True)
        if gt_unique_id[0][0] == "":
            gt_matches = np_matches[gt_unique_id[1][1:]]
        else:
            gt_matches = np_matches[gt_unique_id[1]]

        # 得到gt标签数量
        gt_classes, gt_labels = np.unique(gt_matches[:, 0], return_counts=True)
        for c, n_c in zip(gt_classes, gt_labels):
            ap_dict[c]["标签数量"] = int(n_c)

        # 得到gt类别预测正确数目
        gt_mask = (gt_matches[:, 0] == gt_matches[:, 1])
        for c, n_c in zip(*np.unique(gt_matches[gt_mask, 0], return_counts=True)):
            ap_dict[c]["正确检出"] = int(n_c)

        # 得到gt类别漏检数(其实可以相减, 这里做个验证)
        gt_mask = (gt_matches[:, 0] != gt_matches[:, 1])
        for c, n_c in zip(*np.unique(gt_matches[gt_mask, 0], return_counts=True)):
            ap_dict[c]["漏检数"] = int(n_c)
        
        # 计算recall
        for c in gt_classes:
            ap_dict[c]["recall"] = ap_dict[c]["正确检出"]/ap_dict[c]["标签数量"]

        # 得到平均IOU和CONFIDENCE
        for c in gt_classes:
            ap_dict[c]["平均IOU"], ap_dict[c]["平均CONF"] = np.mean(gt_matches[gt_matches[:, 0] == c, 2:4], axis=0).tolist()
        
        # 得到dt数量
        np_matches = np_matches[np_matches[:, 2].argsort()[::-1]] # IOU排序
        dt_unique_id = np.unique(np_matches[:, 5], return_index=True)
        if dt_unique_id[0][0] == "":
            dt_matches = np_matches[dt_unique_id[1][1:]]
        else:
            dt_matches = np_matches[dt_unique_id[1]]
        
        # 得到dt检出数量
        dt_classes, dt_labels = np.unique(dt_matches[:, 1], return_counts=True)
        dt_labels_dict = {c: int(n_c) for c, n_c in zip(dt_classes, dt_labels)}

        # 得到dt正确检出数量
        dt_mask = (dt_matches[:, 0] == dt_matches[:, 1])
        for c, n_c in zip(*np.unique(dt_matches[dt_mask, 1], return_counts=True)):
            ap_dict[c]["正确预测"] = int(n_c)

        # 得到过杀数目
        dt_mask = ((dt_matches[:, 0] != dt_matches[:, 1]) & (dt_matches[:, 0] == ""))
        for c, n_c in zip(*np.unique(dt_matches[dt_mask, 1], return_counts=True)):
            ap_dict[c]["过杀数"] = int(n_c)

        # 得到误检数目
        dt_mask = ((dt_matches[:, 0] != dt_matches[:, 1]) & (dt_matches[:, 0] != ""))
        for c, n_c in zip(*np.unique(dt_matches[dt_mask, 1], return_counts=True)):
            ap_dict[c]["误检数"] = int(n_c)

        # 得到precision
        for c, total_c in dt_labels_dict.items():
            ap_dict[c]["precision"] = ap_dict[c]["正确预测"]/total_c

    return ap_dict
# Joyphy

class COCOFitowDetectionResults(DetectionResults):
    """Class that stores the results of a COCO detection evaluation.

    Args:
        matches: a list of
            ``(gt_label, pred_label, iou, pred_confidence, gt_id, pred_id)``
            matches. Either label can be ``None`` to indicate an unmatched
            object
        precision: an array of precision values of shape
            ``num_iou_threshs x num_classes x num_recall``
        recall: an array of recall values
        iou_threshs: an array of IoU thresholds
        classes: the list of possible classes
        thresholds (None): an optional array of decision thresholds of shape
            ``num_iou_threshs x num_classes x num_recall``
        eval_key (None): the evaluation key for this evaluation
        gt_field (None): the name of the ground truth field
        pred_field (None): the name of the predictions field
        missing (None): a missing label string. Any unmatched objects are
            given this label for evaluation purposes
        samples (None): the :class:`fiftyone.core.collections.SampleCollection`
            for which the results were computed
    """

    def __init__(
        self,
        matches,
        precision,
        recall,
        iou_threshs,
        classes,
        thresholds=None,
        eval_key=None,
        gt_field=None,
        pred_field=None,
        missing=None,
        samples=None,
        ap_dict=None
    ):
        super().__init__(
            matches,
            eval_key=eval_key,
            gt_field=gt_field,
            pred_field=pred_field,
            classes=classes,
            missing=missing,
            samples=samples,
        )

        self.precision = np.asarray(precision)
        self.recall = np.asarray(recall)
        self.iou_threshs = np.asarray(iou_threshs)
        self.thresholds = (
            np.asarray(thresholds) if thresholds is not None else None
        )

        self._classwise_AP = np.mean(precision, axis=(0, 2))
        self.ap_dict = ap_dict

    # by Joyphy
    def print_report_fitow(self, digits=2, columns_show=None):
        if self.ap_dict == {}:
            print("No classes to analyze")
            return
        else:
            import pandas as pd
            pd.set_option('display.unicode.east_asian_width', True)
            pd.set_option('display.unicode.ambiguous_as_wide', True)
            pd.set_option('display.max_columns',None)
            pd.set_option('display.width', 100000)

            pd_dict = {}
            columns = ['precision', 'recall', 'AP', '平均IOU', '平均CONF', '标签数量', '正确检出', '漏检数', '正确预测', '过杀数', '误检数']
            for c in self.ap_dict:
                pd_dict[c] = [self.ap_dict[c][col] for col in columns]
            pd_df = pd.DataFrame.from_dict(pd_dict, orient="index", columns=columns)
            pd_df[pd_df == -1] = np.NaN
            pd_df.sort_index(inplace=True)
            if columns_show != None:
                pd_df_show = pd_df[columns_show]
            else:
                pd_df_show = pd_df
            results_show = pd_df_show.round(digits)
            print(results_show)
            print("-"*max(map(lambda s: len(s), results_show.__str__().split("\n"))))
            print(f'平均AP: {round(pd_df["AP"].mean(), digits)}')
            print(f'总标签数: {pd_df["标签数量"].sum()}, 总正确检出: {pd_df["正确检出"].sum()}, 平均recall: {round(pd_df["正确检出"].sum()/pd_df["标签数量"].sum(), digits)}, 总漏检数: {pd_df["漏检数"].sum()}')
            print(f'总检出数: {pd_df[["正确预测", "过杀数", "误检数"]].sum().sum()}, 总正确预测: {pd_df["正确预测"].sum()}, 平均precision: {round(pd_df["正确预测"].sum()/pd_df[["正确预测", "过杀数", "误检数"]].sum().sum(), digits)}, 总过杀数: {pd_df["过杀数"].sum()}, 总误检数: {pd_df["误检数"].sum()}')
    # Joyphy

    def plot_pr_curves(
        self, classes=None, iou_thresh=None, backend="plotly", **kwargs
    ):
        """Plots precision-recall (PR) curves for the results.

        Args:
            classes (None): a list of classes to generate curves for. By
                default, the top 3 AP classes will be plotted
            iou_thresh (None): an optional IoU threshold or list of IoU
                thresholds for which to plot curves. If multiple thresholds are
                provided, precision data is averaged across these thresholds.
                By default, precision data is averaged over all IoU thresholds.
                Refer to :attr:`iou_threshs` to see the available thresholds
            backend ("plotly"): the plotting backend to use. Supported values
                are ``("plotly", "matplotlib")``
            **kwargs: keyword arguments for the backend plotting method:

                -   "plotly" backend: :meth:`fiftyone.core.plots.plotly.plot_pr_curves`
                -   "matplotlib" backend: :meth:`fiftyone.core.plots.matplotlib.plot_pr_curves`

        Returns:
            one of the following:

            -   a :class:`fiftyone.core.plots.plotly.PlotlyNotebookPlot`, if
                you are working in a notebook context and the plotly backend is
                used
            -   a plotly or matplotlib figure, otherwise
        """
        if classes is None:
            inds = np.argsort(self._classwise_AP)[::-1]
            classes = self.classes[inds]

        thresh_inds = self._get_iou_thresh_inds(iou_thresh=iou_thresh)

        precisions = []

        has_thresholds = self.thresholds is not None
        thresholds = [] if has_thresholds else None

        for c in classes:
            class_ind = self._get_class_index(c)
            precisions.append(
                np.mean(self.precision[thresh_inds, class_ind], axis=0)
            )
            if has_thresholds:
                thresholds.append(
                    np.mean(self.thresholds[thresh_inds, class_ind], axis=0)
                )

        return fop.plot_pr_curves(
            precisions,
            self.recall,
            classes,
            thresholds=thresholds,
            backend=backend,
            **kwargs,
        )

    def mAP(self, classes=None):
        """Computes COCO-style mean average precision (mAP) for the specified
        classes.

        See `this page <https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py>`_
        for more details about COCO-style mAP.

        Args:
            classes (None): a list of classes for which to compute mAP

        Returns:
            the mAP in ``[0, 1]``
        """
        if classes is not None:
            class_inds = np.array([self._get_class_index(c) for c in classes])
            classwise_AP = self._classwise_AP[class_inds]
        else:
            classwise_AP = self._classwise_AP

        classwise_AP = classwise_AP[classwise_AP > -1]
        if classwise_AP.size == 0:
            return -1

        return np.mean(classwise_AP)

    def _get_iou_thresh_inds(self, iou_thresh=None):
        if iou_thresh is None:
            return np.arange(len(self.iou_threshs))

        if etau.is_numeric(iou_thresh):
            iou_threshs = [iou_thresh]
        else:
            iou_threshs = iou_thresh

        thresh_inds = []
        for iou_thresh in iou_threshs:
            inds = np.where(np.abs(iou_thresh - self.iou_threshs) < 1e-6)[0]
            if inds.size == 0:
                raise ValueError(
                    "Invalid IoU threshold %f. Refer to `results.iou_threshs` "
                    "to see the available values" % iou_thresh
                )

            thresh_inds.append(inds[0])

        return thresh_inds

    def _get_class_index(self, label):
        inds = np.where(self.classes == label)[0]
        if inds.size == 0:
            raise ValueError("Class '%s' not found" % label)

        return inds[0]

    @classmethod
    def _from_dict(cls, d, samples, config, **kwargs):
        precision = d["precision"]
        recall = d["recall"]
        iou_threshs = d["iou_threshs"]
        thresholds = d.get("thresholds", None)
        kwargs["ap_dict"] = d["ap_dict"]
        return super()._from_dict(
            d,
            samples,
            config,
            precision=precision,
            recall=recall,
            iou_threshs=iou_threshs,
            thresholds=thresholds,
            **kwargs,
        )


_NO_MATCH_ID = ""
_NO_MATCH_IOU = None


def _coco_evaluation_single_iou(gts, preds, eval_key, config):
    iou_thresh = min(config.iou, 1 - 1e-10)
    id_key = "%s_id" % eval_key
    iou_key = "%s_iou" % eval_key

    cats, pred_ious, iscrowd = _coco_evaluation_setup(
        gts, preds, [id_key], iou_key, config
    )

    matches = _compute_matches(
        cats,
        pred_ious,
        iou_thresh,
        iscrowd,
        eval_key=eval_key,
        id_key=id_key,
        iou_key=iou_key,
    )

    # omit iscrowd
    return [m[:-1] for m in matches]

# by Joyphy
def _fo_to_numpy(targets, classes, mode=5, dtype=np.float32):
    """
    fiftyone Detections转成numpy数据接口.
    mode: 5表示标签, 6表示预测
    """
    if mode not in (5, 6):
        raise ValueError(f"mode只能是5, 6; 但是收到值{mode}")

    if targets == None:
        return np.zeros([0, mode], dtype=dtype)
    else:
        if mode == 6:
            np_value =np.array(
                [[*d["bounding_box"], d["confidence"], classes[d["label"]]] for d in targets["detections"]],
                dtype=dtype)
            np_value[:, 2:4] = np_value[:, 0:2] + np_value[:, 2:4]
        elif mode == 5:
            np_value =np.array(
                [[classes[d["label"]], *d["bounding_box"]] for d in targets["detections"]],
                dtype=dtype)
            np_value[:, 3:5] = np_value[:, 1:3] + np_value[:, 3:5]
        return np_value

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def _coco_evaluation_single_iou_numpy(gts, preds, eval_key, config):
    matches_return = []

    iou_thresh = min(config.iou, 1 - 1e-10)
    id_key = "%s_id" % eval_key
    id_keys = [id_key]
    iou_key = "%s_iou" % eval_key

    if gts is None:
        gts = {"detections": []}
    if preds is None:
        preds = {"detections": []}
    classes = {name: i for i, name in enumerate({d["label"] for d in gts["detections"]}.union({d["label"] for d in preds["detections"]}))}

    if len(gts["detections"]):
        for obj in gts[gts._LABEL_LIST_FIELD]:
            obj[iou_key] = _NO_MATCH_IOU
            for id_key in id_keys:
                obj[id_key] = _NO_MATCH_ID
        labelsn = _fo_to_numpy(gts, classes=classes, mode=5, dtype=np.float64)
    else:
        labelsn = np.zeros((0, 5), dtype=np.float64)
    if len(preds["detections"]):
        for obj in preds[preds._LABEL_LIST_FIELD]:
            obj[iou_key] = _NO_MATCH_IOU
            for id_key in id_keys:
                obj[id_key] = _NO_MATCH_ID
        predsn = _fo_to_numpy(preds, classes=classes, mode=6, dtype=np.float64)
    else:
        predsn = np.zeros((0, 6), dtype=np.float64)

    iou = box_iou(labelsn[:, 1:], predsn[:, :4])
    iou_thresh = min(config.iou, 1 - 1e-10)

    x = np.where((iou >= iou_thresh) & (labelsn[:, 0:1] == predsn[:, 5]))

    matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None], predsn[x[1], 4:5]), 1)  # [label, detection, iou, confidence]
    matches_ori = matches

    # n, 1表示每个dt或gt能用n次或只能用1次
    if config.eval_mode == "nvn":
        matches = matches[matches[:, 3].argsort()[::-1]] # 置信度排序
        gt_tp = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]] # iou排序
        dt_tp = matches[np.unique(matches[:, 1], return_index=True)[1]]

    elif config.eval_mode == "nv1":
        matches = matches[matches[:, 2].argsort()[::-1]] # iou排序
        dt_tp = matches[np.unique(matches[:, 1], return_index=True)[1]]
        # 消除多个gt匹配一个dt, dt算iou最高的, 此举会降低Recall
        matches = matches[sorted(np.unique(matches[:, 1], return_index=True)[1])]
        matches = matches[matches[:, 3].argsort()[::-1]] # 置信度排序
        gt_tp = matches[np.unique(matches[:, 0], return_index=True)[1]]

    elif config.eval_mode == "1vn":
        matches = matches[matches[:, 3].argsort()[::-1]] # 置信度排序
        gt_tp = matches[np.unique(matches[:, 0], return_index=True)[1]]
        # 消除多个dt匹配一个gt, gt算置信度最高的, 此举会降低Precision
        matches = matches[sorted(np.unique(matches[:, 0], return_index=True)[1])]
        matches = matches[matches[:, 2].argsort()[::-1]] # iou排序
        dt_tp = matches[np.unique(matches[:, 1], return_index=True)[1]]

    elif config.eval_mode == "1v1":
        matches = matches[matches[:, 2].argsort()[::-1]] # iou排序
        # 消除多个gt匹配一个dt, dt算iou最高的, 此举会降低Recall
        matches = matches[sorted(np.unique(matches[:, 1], return_index=True)[1])]
        matches = matches[matches[:, 3].argsort()[::-1]] # 置信度排序
        # 消除多个dt匹配一个gt, gt算置信度最高的, 此举会降低Precision
        matches = matches[sorted(np.unique(matches[:, 0], return_index=True)[1])]
        gt_tp = matches[np.unique(matches[:, 0], return_index=True)[1]]
        dt_tp = matches[np.unique(matches[:, 1], return_index=True)[1]]

    else:
        raise ValueError(f"eval_mode只支持1v1, 1vn, nv1, nvn")

    gt_fn  = np.setdiff1d([i for i in range(labelsn.shape[0])], gt_tp[:, 0])
    dt_fp_gs = np.setdiff1d(matches_ori[:, 1], dt_tp[:, 1])
    dt_fp = np.setdiff1d([i for i in range(predsn.shape[0])], matches_ori[:, 1])
    if dt_fp.shape[0]:
        wj_x = np.where((iou[:, dt_fp] >= iou_thresh) & (labelsn[:, 0:1] != predsn[dt_fp, 5]))
        if wj_x[0].shape[0]:
            wj_matches = np.concatenate((wj_x[0][:, None], dt_fp[wj_x[1]][:, None], iou[wj_x[0], dt_fp[wj_x[1]]][:, None]), 1)
            wj_matches = wj_matches[wj_matches[:, 2].argsort()[::-1]] # iou排序
            dt_fp_wj = wj_matches[np.unique(wj_matches[:, 1], return_index=True)[1]]
            dt_fp_gs = np.union1d(np.setdiff1d(dt_fp, dt_fp_wj[:, 1]), dt_fp_gs)
            # dt_fp_wj
            for gi, pi, match_iou in dt_fp_wj[:, 0:3].tolist():
                gt = gts["detections"][int(gi)]
                pred = preds["detections"][int(pi)]

                pred[eval_key] = "fp_wj"
                pred[id_key] = gt.id
                pred[iou_key] = match_iou
                matches_return.append(
                    (
                        gt.label,
                        pred.label,
                        match_iou,
                        pred.confidence,
                        gt.id,
                        pred.id
                    )
                )
        else:
            dt_fp_gs = np.union1d(dt_fp, dt_fp_gs)

    # gt_tp
    for gi, pi, match_iou in gt_tp[:, 0:3].tolist():
        gt = gts["detections"][int(gi)]
        pred = preds["detections"][int(pi)]

        gt[eval_key] = "tp"
        gt[id_key] = pred.id
        gt[iou_key] = match_iou
        matches_return.append(
            (
                gt.label,
                pred.label,
                match_iou,
                pred.confidence,
                gt.id,
                pred.id
            )
        )

    # gt_fn
    for gi in gt_fn:
        gt = gts["detections"][int(gi)]
        gt[eval_key] = "fn"
        matches_return.append(
            (gt.label, None, None, None, gt.id, None)
        )

    # dt_tp
    over_indexes = gt_tp[:, 0:2].tolist()
    for gi, pi, match_iou in dt_tp[:, 0:3].tolist():
        gt = gts["detections"][int(gi)]
        pred = preds["detections"][int(pi)]

        pred[eval_key] = "tp"
        pred[id_key] = gt.id
        pred[iou_key] = match_iou

        if [gi, pi] in over_indexes:
            continue

        matches_return.append(
            (
                gt.label,
                pred.label,
                match_iou,
                pred.confidence,
                gt.id,
                pred.id
            )
        )

    # dt_fp_gs
    for pi in dt_fp_gs:
        pred = preds["detections"][int(pi)]
        pred[eval_key] = "fp_gs"
        matches_return.append(
            (None, pred.label, None, pred.confidence, None, pred.id)
        )

    return matches_return
# Joyphy

def _coco_evaluation_iou_sweep(gts, preds, config):
    iou_threshs = config.iou_threshs
    id_keys = ["eval_id_%s" % str(i).replace(".", "_") for i in iou_threshs]
    iou_key = "eval_iou"

    cats, pred_ious, iscrowd = _coco_evaluation_setup(
        gts, preds, id_keys, iou_key, config, max_preds=config.max_preds
    )

    return [
        _compute_matches(
            cats,
            pred_ious,
            iou_thresh,
            iscrowd,
            eval_key="_eval",
            id_key=id_key,
            iou_key=iou_key,
        )
        for iou_thresh, id_key in zip(iou_threshs, id_keys)
    ]


def _coco_evaluation_setup(
    gts, preds, id_keys, iou_key, config, max_preds=None
):
    iscrowd = lambda l: bool(l.get_attribute_value(config.iscrowd, False))
    classwise = config.classwise

    iou_kwargs = dict(iscrowd=iscrowd, error_level=config.error_level)

    if config.use_masks:
        iou_kwargs.update(use_masks=True, tolerance=config.tolerance)

    if config.use_boxes:
        iou_kwargs.update(use_boxes=True)

    # Organize ground truth and predictions by category

    cats = defaultdict(lambda: defaultdict(list))

    if gts is not None:
        for obj in gts[gts._LABEL_LIST_FIELD]:
            obj[iou_key] = _NO_MATCH_IOU
            for id_key in id_keys:
                obj[id_key] = _NO_MATCH_ID

            label = obj.label if classwise else "all"
            cats[label]["gts"].append(obj)

    if preds is not None:
        for obj in preds[preds._LABEL_LIST_FIELD]:
            obj[iou_key] = _NO_MATCH_IOU
            for id_key in id_keys:
                obj[id_key] = _NO_MATCH_ID

            label = obj.label if classwise else "all"
            cats[label]["preds"].append(obj)

    # Compute IoUs within each category
    pred_ious = {}
    for objects in cats.values():
        gts = objects["gts"]
        preds = objects["preds"]

        # Highest confidence predictions first
        preds = sorted(preds, key=lambda p: p.confidence or -1, reverse=True)

        if max_preds is not None:
            preds = preds[:max_preds]

        objects["preds"] = preds

        # Sort ground truth so crowds are last
        gts = sorted(gts, key=iscrowd)

        # Compute ``num_preds x num_gts`` IoUs
        ious = foui.compute_ious(preds, gts, **iou_kwargs)

        gt_ids = [g.id for g in gts]
        for pred, gt_ious in zip(preds, ious):
            pred_ious[pred.id] = list(zip(gt_ids, gt_ious))

    return cats, pred_ious, iscrowd


def _compute_matches(
    cats, pred_ious, iou_thresh, iscrowd, eval_key, id_key, iou_key
):
    matches = []

    # Match preds to GT, highest confidence first
    for cat, objects in cats.items():
        gt_map = {gt.id: gt for gt in objects["gts"]}

        # Match each prediction to the highest available IoU ground truth
        for pred in objects["preds"]:
            if pred.id in pred_ious:
                best_match = None
                best_match_iou = iou_thresh
                for gt_id, iou in pred_ious[pred.id]:
                    gt = gt_map[gt_id]
                    gt_iscrowd = iscrowd(gt)

                    # Only iscrowd GTs can have multiple matches
                    if gt[id_key] != _NO_MATCH_ID and not gt_iscrowd:
                        continue

                    # If matching classwise=False
                    # Only objects with the same class can match a crowd
                    if gt_iscrowd and gt.label != pred.label:
                        continue

                    # Crowds are last in order of GTs
                    # If we already matched a non-crowd and are on a crowd,
                    # then break
                    if (
                        best_match
                        and not iscrowd(gt_map[best_match])
                        and gt_iscrowd
                    ):
                        break

                    if iou < best_match_iou:
                        continue

                    best_match_iou = iou
                    best_match = gt_id

                if best_match:
                    gt = gt_map[best_match]

                    # For crowd GTs, record info for first (highest confidence)
                    # matching prediction on the GT object
                    if gt[id_key] == _NO_MATCH_ID:
                        gt[eval_key] = "tp" if gt.label == pred.label else "fn"
                        gt[id_key] = pred.id
                        gt[iou_key] = best_match_iou

                    pred[eval_key] = "tp" if gt.label == pred.label else "fp"
                    pred[id_key] = best_match
                    pred[iou_key] = best_match_iou

                    matches.append(
                        (
                            gt.label,
                            pred.label,
                            best_match_iou,
                            pred.confidence,
                            gt.id,
                            pred.id,
                            iscrowd(gt),
                        )
                    )
                else:
                    pred[eval_key] = "fp"
                    matches.append(
                        (
                            None,
                            pred.label,
                            None,
                            pred.confidence,
                            None,
                            pred.id,
                            None,
                        )
                    )

            elif pred.label == cat:
                pred[eval_key] = "fp"
                matches.append(
                    (
                        None,
                        pred.label,
                        None,
                        pred.confidence,
                        None,
                        pred.id,
                        None,
                    )
                )

        # Leftover GTs are false negatives
        for gt in objects["gts"]:
            if gt[id_key] == _NO_MATCH_ID:
                gt[eval_key] = "fn"
                matches.append(
                    (gt.label, None, None, None, gt.id, None, iscrowd(gt))
                )

    return matches

def _compute_pr_curves(samples, config, classes=None):
    gt_field = config.gt_field
    pred_field = config.pred_field
    iou_threshs = config.iou_threshs

    samples = samples.select_fields([gt_field, pred_field])

    gt_field, processing_frames = samples._handle_frame_field(gt_field)
    pred_field, _ = samples._handle_frame_field(pred_field)

    num_threshs = len(iou_threshs)
    thresh_matches = [{} for _ in range(num_threshs)]

    if classes is None:
        _classes = set()

    logger.info("Performing IoU sweep...")
    for sample in samples.iter_samples(progress=True):
        if processing_frames:
            images = sample.frames.values()
        else:
            images = [sample]

        for image in images:
            # Don't edit user's data during sweep
            gts = _copy_labels(image[gt_field])
            preds = _copy_labels(image[pred_field])

            matches_list = _coco_evaluation_iou_sweep(gts, preds, config)

            for idx, matches in enumerate(matches_list):
                for match in matches:
                    gt_label = match[0]
                    pred_label = match[1]
                    iscrowd = match[-1]

                    if classes is None:
                        _classes.add(gt_label)
                        _classes.add(pred_label)

                    if iscrowd:
                        continue

                    c = gt_label if gt_label is not None else pred_label

                    if c not in thresh_matches[idx]:
                        thresh_matches[idx][c] = {
                            "tp": [],
                            "fp": [],
                            "num_gt": 0,
                        }

                    if gt_label == pred_label:
                        thresh_matches[idx][c]["tp"].append(match)
                    elif pred_label:
                        thresh_matches[idx][c]["fp"].append(match)

                    if gt_label:
                        thresh_matches[idx][c]["num_gt"] += 1

    if classes is None:
        _classes.discard(None)
        classes = sorted(_classes)

    num_classes = len(classes)
    class_idx_map = {c: idx for idx, c in enumerate(classes)}

    # Compute precision-recall
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    precision = -np.ones((num_threshs, num_classes, 101))
    thresholds = -np.ones((num_threshs, num_classes, 101))
    recall = np.linspace(0, 1, 101)
    for idx, _thresh_matches in enumerate(thresh_matches):
        for c, matches in _thresh_matches.items():
            c_idx = class_idx_map.get(c, None)
            num_gt = matches["num_gt"]

            if c_idx is None or num_gt == 0:
                continue

            tp = matches["tp"]
            fp = matches["fp"]
            tp_fp = np.array([1] * len(tp) + [0] * len(fp))
            confs = np.array([m[3] for m in tp] + [m[3] for m in fp])
            if None in confs:
                raise ValueError(
                    "All predicted objects must have their `confidence` "
                    "attribute populated in order to compute precision-recall "
                    "curves"
                )

            inds = np.argsort(-confs, kind="mergesort")
            tp_fp = tp_fp[inds]
            confs = confs[inds]

            tp_sum = np.cumsum(tp_fp).astype(dtype=float)
            total = np.arange(1, len(tp_fp) + 1).astype(dtype=float)

            pre = tp_sum / total
            rec = tp_sum / num_gt

            for i in range(len(pre) - 1, 0, -1):
                if pre[i] > pre[i - 1]:
                    pre[i - 1] = pre[i]

            q = np.zeros(101)
            t = np.zeros(101)

            inds = np.searchsorted(rec, recall, side="left")

            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pre[pi]
                    t[ri] = confs[pi]
            except:
                pass

            precision[idx][c_idx] = q
            thresholds[idx][c_idx] = t

    return precision, recall, thresholds, iou_threshs, classes

def _copy_labels(labels):
    if labels is None:
        return None

    field = labels._LABEL_LIST_FIELD
    _labels = labels.copy()

    # We need the IDs to stay the same
    for _label, label in zip(_labels[field], labels[field]):
        _label.id = label.id

    return _labels
