"""
评估模块

提供题目提取结果的评估功能
"""
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ExtractionEvaluator:
    """题目提取评估器"""

    def evaluate(
        self,
        extracted_questions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        评估提取结果

        指标：
        1. 题目检测准确率：正确识别的题目数 / 总题目数
        2. 边界框IoU：预测框与真实框的交并比
        3. 图文关联准确率：正确关联的图像数 / 总图像数

        Args:
            extracted_questions: 提取的题目列表
            ground_truth: 标注的真实题目列表

        Returns:
            评估指标字典
        """
        logger.info("开始评估提取结果")

        metrics = {
            'detection_precision': 0.0,  # 检测精确率
            'detection_recall': 0.0,     # 检测召回率
            'detection_f1': 0.0,         # F1分数
            'bbox_iou_mean': 0.0,        # 平均IoU
            'image_association_acc': 0.0, # 图像关联准确率
            'split_accuracy': 0.0,       # 拆分准确率
        }

        # 1. 题目检测评估
        detected_ids = set(q['id'] for q in extracted_questions if q.get('id'))
        ground_truth_ids = set(q['id'] for q in ground_truth if q.get('id'))

        tp = len(detected_ids & ground_truth_ids)  # 真阳性
        fp = len(detected_ids - ground_truth_ids)  # 假阳性
        fn = len(ground_truth_ids - detected_ids)  # 假阴性

        metrics['detection_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['detection_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        if metrics['detection_precision'] + metrics['detection_recall'] > 0:
            metrics['detection_f1'] = (
                2 * metrics['detection_precision'] * metrics['detection_recall'] /
                (metrics['detection_precision'] + metrics['detection_recall'])
            )

        # 2. 边界框评估
        ious = []
        for extracted in extracted_questions:
            for gt in ground_truth:
                if extracted.get('id') == gt.get('id'):
                    iou = self._calculate_iou(
                        extracted.get('bbox', []),
                        gt.get('bbox', [])
                    )
                    ious.append(iou)
                    break

        metrics['bbox_iou_mean'] = sum(ious) / len(ious) if ious else 0

        # 3. 图像关联评估
        image_correct = 0
        image_total = 0

        for extracted in extracted_questions:
            for gt in ground_truth:
                if extracted.get('id') == gt.get('id'):
                    extracted_images = set(
                        img['block_id'] for img in extracted.get('content', {}).get('images', [])
                    )
                    gt_images = set(
                        img.get('block_id') for img in gt.get('content', {}).get('images', [])
                    )

                    if extracted_images == gt_images:
                        image_correct += 1
                    image_total += 1
                    break

        metrics['image_association_acc'] = (
            image_correct / image_total if image_total > 0 else 0
        )

        # 4. 拆分准确率评估
        split_correct = 0
        split_total = 0

        for extracted in extracted_questions:
            is_split = extracted.get('metadata', {}).get('split_from_merged', False)
            if is_split:
                split_total += 1

                # 检查是否正确拆分（边界框准确）
                for gt in ground_truth:
                    if extracted.get('id') == gt.get('id'):
                        iou = self._calculate_iou(
                            extracted.get('bbox', []),
                            gt.get('bbox', [])
                        )
                        if iou > 0.7:  # IoU阈值
                            split_correct += 1
                        break

        metrics['split_accuracy'] = (
            split_correct / split_total if split_total > 0 else 0
        )

        # 打印评估结果
        logger.info("=== 评估结果 ===")
        logger.info(f"检测精确率: {metrics['detection_precision']:.2%}")
        logger.info(f"检测召回率: {metrics['detection_recall']:.2%}")
        logger.info(f"检测F1分数: {metrics['detection_f1']:.2%}")
        logger.info(f"平均IoU: {metrics['bbox_iou_mean']:.2%}")
        logger.info(f"图像关联准确率: {metrics['image_association_acc']:.2%}")
        logger.info(f"拆分准确率: {metrics['split_accuracy']:.2%}")

        return metrics

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        计算两个边界框的IoU（交并比）

        Args:
            bbox1, bbox2: [x1, y1, x2, y2]

        Returns:
            IoU值 (0.0-1.0)
        """
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
