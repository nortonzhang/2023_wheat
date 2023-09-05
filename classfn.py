from skimage import measure
from scipy import ndimage
import cv2 as cv
import numpy as np


def get_buildings(mask, pixel_threshold):
    gt_labeled_array, gt_num = ndimage.label(mask)
    unique, counts = np.unique(gt_labeled_array, return_counts=True)
    for (k, v) in dict(zip(unique, counts)).items():
        if v < pixel_threshold:
            mask[gt_labeled_array == k] = 0
    return measure.label(mask, return_num=True)


def calculate_f1_buildings_score(y_pred_path, iou_threshold=0.40, component_size_threshold=0):
    # iou_threshold=0.40表示重合面积大于40%，判断为TP
    tp = 0
    fp = 0
    fn = 0

    # for m in tqdm(range(len(y_pred_list))):
    processed_gt = set()
    matched = set()

    # mask_img是预测图像
    # mask_img = cv.imread(r".\predictLabel\Halo-water.jpg", 0)
    # mask_img = cv.imread(r".\predictLabel\Halo_image.png", 0)
    mask_img = cv.imread(r".\predictLabel\RGB_image.png", 0)
    # gt_mask_img 是groundTruth图像
    gt_mask_img = cv.imread(r".\groundtruth\GT_image.png", 0)

    predicted_labels, predicted_count = get_buildings(mask_img, component_size_threshold)
    gt_labels, gt_count = get_buildings(gt_mask_img, component_size_threshold)

    gt_buildings = [rp.coords for rp in measure.regionprops(gt_labels)]
    pred_buildings = [rp.coords for rp in measure.regionprops(predicted_labels)]
    gt_buildings = [to_point_set(b) for b in gt_buildings]
    pred_buildings = [to_point_set(b) for b in pred_buildings]
    for j in range(predicted_count):
        match_found = False
        for i in range(gt_count):
            pred_ind = j + 1
            gt_ind = i + 1
            if match_found:
                break
            if gt_ind in processed_gt:
                continue
            pred_building = pred_buildings[j]
            gt_building = gt_buildings[i]
            intersection = len(pred_building.intersection(gt_building))
            union = len(pred_building) + len(gt_building) - intersection
            iou = intersection / union
            if iou > iou_threshold:
                processed_gt.add(gt_ind)
                matched.add(pred_ind)
                match_found = True
                tp += 1
        if not match_found:
            fp += 1
    fn += gt_count - len(processed_gt)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f_score = 2 * precision * recall / (precision + recall)
    return f_score, fp, fn, tp, precision, recall


def to_point_set(building):
    return set([(row[0], row[1]) for row in building])


# y_pred_path 没用到，随便填
y_pred_path = 'predictLabel'

f_score = calculate_f1_buildings_score(y_pred_path, iou_threshold=0.5, component_size_threshold=1)

print(f_score)
