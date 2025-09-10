import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

def evaluate_segmentation_folder(pred_dir, gt_dir, threshold=128):
    """
    批量计算并返回整个文件夹中道路提取的各项评价指标。

    参数:
    - pred_dir (str): 预测结果掩码图像所在的文件夹路径。
    - gt_dir (str): 真值标签掩码图像所在的文件夹路径。
    - threshold (int): 将灰度图转换为二值图的阈值。

    返回:
    - dict: 包含所有计算出的平均指标的字典。
    """
    if not os.path.isdir(pred_dir):
        raise NotADirectoryError(f"预测文件夹不存在: {pred_dir}")
    if not os.path.isdir(gt_dir):
        raise NotADirectoryError(f"真值文件夹不存在: {gt_dir}")

    # 获取预测文件夹中的所有图像文件名
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    
    if not pred_files:
        print(f"警告: 在 '{pred_dir}' 中没有找到任何图像文件。")
        return None

    # 初始化用于累加的统计量
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

    # 使用 tqdm 创建进度条
    for filename in tqdm(pred_files, desc="正在处理图像"):
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename.replace('.jpg','.png'))

        if not os.path.exists(gt_path):
            print(f"警告: 找不到对应的真值文件 '{filename}'，已跳过。")
            continue

        pred_img = cv2.imread(pred_path)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred_img is None or gt_img is None:
            print(f"警告: 无法读取图像 '{filename}'，已跳过。")
            continue

        if pred_img.shape != gt_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # pred_mask = pred_img > threshold
        # gt_mask = gt_img > threshold
        pred_mask = pred_img[:,:,0] > threshold
        gt_mask = gt_img.astype(bool)

        total_tp += np.sum(np.logical_and(pred_mask, gt_mask))
        total_fp += np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))
        total_tn += np.sum(np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)))
        total_fn += np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))

    # --- 基于累加的总数计算最终指标 ---
    epsilon = 1e-10

    # IoU for Road Class
    iou_road = total_tp / (total_tp + total_fp + total_fn + epsilon)

    # IoU for Background Class
    iou_background = total_tn / (total_tn + total_fn + total_fp + epsilon)
    
    # MIoU (Mean IoU)
    miou = (iou_road + iou_background) / 2

    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + epsilon)
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    metrics = {
        'MIoU': miou,
        'IoU (Road)': iou_road,
        'Accuracy': accuracy,
        'F1-score': f1_score,
        'Precision': precision,
        'Recall': recall,
        'Total TP': int(total_tp),
        'Total FP': int(total_fp),
        'Total TN': int(total_tn),
        'Total FN': int(total_fn)
    }
    
    return metrics

def print_metrics(metrics):
    """格式化打印指标"""
    print("\n--- 道路提取批量评估结果 ---")
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:<12}: {value:.4f}")
            else:
                print(f"{key:<12}: {value}")
    else:
        print("没有计算任何指标。")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量计算道路提取的分割评价指标。')
    parser.add_argument('--pred_dir', type=str,default='./deep/train/vis', help='包含预测结果掩码图像的文件夹路径。')
    parser.add_argument('--gt_dir', type=str,default='/root/autodl-tmp/roaddataset/deepglobe/annotations/train', help='包含真值标签掩码图像的文件夹路径。')
    parser.add_argument('--threshold', type=int, default=128, help='用于二值化的像素阈值 (0-255)，默认为128。')

    args = parser.parse_args()

    try:
        results = evaluate_segmentation_folder(args.pred_dir, args.gt_dir, args.threshold)
        print_metrics(results)
    except (NotADirectoryError, Exception) as e:
        print(f"发生错误: {e}")