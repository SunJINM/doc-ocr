"""
结果可视化模块

提供题目提取结果的可视化功能
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """结果可视化工具"""

    def __init__(self):
        """初始化可视化工具"""
        # 颜色定义 (BGR格式)
        self.colors = {
            'text_with_image': (0, 255, 0),     # 绿色：带配图的题目
            'text_no_image': (255, 0, 0),       # 蓝色：无配图的题目
            'image': (0, 255, 255),             # 黄色：图像
            'connection': (255, 255, 0),        # 青色：关联线
            'split': (0, 165, 255),             # 橙色：拆分的题目
        }

    def visualize_extraction_results(
        self,
        original_image: np.ndarray,
        extracted_questions: List[Dict[str, Any]],
        output_path: str = 'output/extraction_visualization.jpg'
    ) -> np.ndarray:
        """
        可视化题目提取结果

        功能：
        1. 在原图上绘制题目边界框
        2. 标注题号和类型
        3. 用不同颜色区分有无配图
        4. 显示图文关联线
        5. 标记拆分的题目

        Args:
            original_image: 原始图像
            extracted_questions: 提取的题目列表
            output_path: 输出路径

        Returns:
            可视化后的图像
        """
        logger.info("开始生成可视化结果")

        vis_image = original_image.copy()

        # 统计信息
        total = len(extracted_questions)
        with_images = 0
        split_count = 0
        total_images = 0

        # 绘制每个题目
        for question in extracted_questions:
            bbox = question.get('bbox', [])
            question_id = question.get('id')
            has_image = question.get('content', {}).get('has_image', False)
            images = question.get('content', {}).get('images', [])
            is_split = question.get('metadata', {}).get('split_from_merged', False)
            question_type = question.get('type', '')

            if has_image:
                with_images += 1
                total_images += len(images)

            if is_split:
                split_count += 1

            # 选择颜色
            if is_split:
                color = self.colors['split']
            elif has_image:
                color = self.colors['text_with_image']
            else:
                color = self.colors['text_no_image']

            # 绘制题目边界框
            if len(bbox) == 4:
                cv2.rectangle(
                    vis_image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    3
                )

                # 标注题号和类型
                label = f"Q{question_id} [{question_type}]"
                if is_split:
                    label += " *Split*"

                # 绘制文本背景
                (text_w, text_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                cv2.rectangle(
                    vis_image,
                    (int(bbox[0]), int(bbox[1]) - text_h - 10),
                    (int(bbox[0]) + text_w + 10, int(bbox[1])),
                    color,
                    -1
                )

                # 绘制文本
                cv2.putText(
                    vis_image,
                    label,
                    (int(bbox[0]) + 5, int(bbox[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # 绘制配图及关联线
            if has_image and images:
                text_center = (
                    int((bbox[0] + bbox[2]) / 2),
                    int((bbox[1] + bbox[3]) / 2)
                )

                for img_info in images:
                    img_bbox = img_info.get('bbox', [])

                    if len(img_bbox) == 4:
                        # 绘制图像框
                        cv2.rectangle(
                            vis_image,
                            (int(img_bbox[0]), int(img_bbox[1])),
                            (int(img_bbox[2]), int(img_bbox[3])),
                            self.colors['image'],
                            2
                        )

                        # 绘制关联线
                        img_center = (
                            int((img_bbox[0] + img_bbox[2]) / 2),
                            int((img_bbox[1] + img_bbox[3]) / 2)
                        )

                        cv2.line(
                            vis_image,
                            text_center,
                            img_center,
                            self.colors['connection'],
                            2,
                            cv2.LINE_AA
                        )

                        # 标记VL验证
                        if img_info.get('vl_verified'):
                            cv2.circle(
                                vis_image,
                                img_center,
                                10,
                                (0, 0, 255),  # 红色圆点表示VL验证通过
                                -1
                            )

        # 添加统计信息
        self._add_statistics_panel(
            vis_image,
            total,
            with_images,
            total_images,
            split_count
        )

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_image)

        logger.info(f"=== 可视化结果统计 ===")
        logger.info(f"总题目数: {total}")
        logger.info(f"带配图题目: {with_images}")
        logger.info(f"总图片数: {total_images}")
        logger.info(f"拆分题目数: {split_count}")
        logger.info(f"可视化图片已保存: {output_path}")

        return vis_image

    def _add_statistics_panel(
        self,
        image: np.ndarray,
        total: int,
        with_images: int,
        total_images: int,
        split_count: int
    ):
        """在图像右上角添加统计信息面板"""
        h, w = image.shape[:2]

        # 面板位置和大小
        panel_width = 300
        panel_height = 180
        panel_x = w - panel_width - 20
        panel_y = 20

        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # 绘制边框
        cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (255, 255, 255),
            2
        )

        # 绘制标题
        cv2.putText(
            image,
            "Extraction Statistics",
            (panel_x + 10, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # 绘制统计信息
        stats = [
            f"Total Questions: {total}",
            f"With Images: {with_images}",
            f"Total Images: {total_images}",
            f"Split Questions: {split_count}"
        ]

        y_offset = 60
        for stat in stats:
            cv2.putText(
                image,
                stat,
                (panel_x + 10, panel_y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 25

    def create_comparison_view(
        self,
        original_image: np.ndarray,
        before_questions: List[Dict[str, Any]],
        after_questions: List[Dict[str, Any]],
        output_path: str = 'output/comparison.jpg'
    ):
        """
        创建优化前后对比视图

        Args:
            original_image: 原始图像
            before_questions: 优化前的题目（PP-OCR-VL原始结果）
            after_questions: 优化后的题目
            output_path: 输出路径
        """
        # 创建两张可视化图像
        before_vis = original_image.copy()
        after_vis = original_image.copy()

        # 绘制优化前（简单绘制原始文本块）
        for q in before_questions:
            bbox = q.get('bbox', [])
            if len(bbox) == 4:
                cv2.rectangle(
                    before_vis,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 0, 255),  # 红色
                    2
                )

        # 绘制优化后
        for q in after_questions:
            bbox = q.get('bbox', [])
            if len(bbox) == 4:
                has_image = q.get('content', {}).get('has_image', False)
                color = self.colors['text_with_image'] if has_image else self.colors['text_no_image']

                cv2.rectangle(
                    after_vis,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2
                )

        # 拼接左右对比
        h, w = original_image.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = before_vis
        comparison[:, w:] = after_vis

        # 添加标题
        cv2.putText(
            comparison,
            "Before Optimization",
            (w // 2 - 150, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

        cv2.putText(
            comparison,
            "After Optimization",
            (w + w // 2 - 150, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, comparison)

        logger.info(f"对比视图已保存: {output_path}")
