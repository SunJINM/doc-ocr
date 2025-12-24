"""
题目拆分模块

处理一个文本块包含多个题目的情况，通过题号识别和OCR定位进行智能拆分
"""
import re
import logging
import tempfile
import os
import json
from typing import List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class QuestionSplitter:
    """题目拆分处理器"""

    def __init__(self, ocr_model, config):
        """
        初始化题目拆分器

        Args:
            ocr_model: PaddleOCR实例
            config: ProcessingConfig配置实例
        """
        self.ocr_model = ocr_model
        self.config = config
        self.question_patterns = config.question_number_patterns

    def split_merged_questions(
        self,
        text_block: Dict[str, Any],
        original_image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        拆分包含多个题目的文本块

        Args:
            text_block: 文本块数据 {'block_content', 'block_bbox', ...}
            original_image: 原始图像数组

        Returns:
            拆分后的独立题目列表
        """
        content = text_block.get('block_content', '')
        bbox = text_block.get('block_bbox', [])

        if not content or not bbox:
            logger.warning(f"文本块内容或坐标为空，跳过拆分")
            return [text_block]

        # 步骤1: 检测题号
        question_numbers = self._detect_question_numbers(content)

        if len(question_numbers) <= 1:
            logger.debug(f"文本块只包含{len(question_numbers)}个题号，不需要拆分")
            # 添加题号信息
            if question_numbers:
                text_block['question_number'] = question_numbers[0]['number']
            return [text_block]

        logger.info(f"检测到{len(question_numbers)}个题号，开始拆分: {[q['number'] for q in question_numbers]}")

        # 步骤2: OCR精细定位
        try:
            question_positions = self._locate_question_numbers_with_ocr(
                question_numbers,
                bbox,
                original_image
            )
        except Exception as e:
            logger.error(f"OCR定位失败: {e}，使用文本位置估算")
            question_positions = self._estimate_positions_from_text(
                question_numbers,
                bbox,
                content
            )

        # 步骤3: 根据位置拆分
        split_questions = self._split_by_positions(
            text_block,
            question_positions,
            question_numbers
        )

        logger.info(f"拆分完成，得到{len(split_questions)}个独立题目")
        return split_questions

    def _detect_question_numbers(self, text: str) -> List[Dict[str, Any]]:
        """
        检测文本中的所有题号

        Returns:
            [{'number': int, 'position': int, 'matched_str': str, 'type': str}]
        """
        question_numbers = []

        for pattern in self.question_patterns:
            try:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    number = int(match.group(1))
                    position = match.start()
                    matched_str = match.group(0)

                    question_numbers.append({
                        'number': number,
                        'position': position,
                        'matched_str': matched_str,
                        'pattern': pattern
                    })
            except Exception as e:
                logger.warning(f"模式匹配失败 {pattern}: {e}")
                continue

        # 按位置排序
        question_numbers.sort(key=lambda x: x['position'])

        # 过滤重复（同一位置可能被多个模式匹配）
        filtered = []
        last_pos = -10
        for qn in question_numbers:
            if qn['position'] - last_pos > 5:
                filtered.append(qn)
                last_pos = qn['position']

        # 验证题号的合理性
        validated = self._validate_question_sequence(filtered, text)

        return validated

    def _validate_question_sequence(
        self,
        question_numbers: List[Dict[str, Any]],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        验证题号序列的合理性

        过滤掉不太可能是题号的匹配（如题目内容中的数字）
        """
        if not question_numbers:
            return []

        validated = []

        for qn in question_numbers:
            # 检查1: 位置应该在行首或接近行首
            if not self._is_at_line_start(qn['position'], text):
                logger.debug(f"题号{qn['number']}不在行首，可能是误匹配")
                continue

            # 检查2: 后续应该有题目内容
            following_text = text[qn['position'] + len(qn['matched_str']):qn['position'] + 100]
            if len(following_text.strip()) < 5:
                logger.debug(f"题号{qn['number']}后续内容太少")
                continue

            validated.append(qn)

        return validated

    def _is_at_line_start(self, position: int, text: str) -> bool:
        """检查位置是否在行首或接近行首（允许前面有少量空格）"""
        if position == 0:
            return True

        # 查找前一个换行符
        before_text = text[:position]
        last_newline = before_text.rfind('\n')

        if last_newline == -1:
            # 没有换行符，检查是否在文本开头附近
            return position < 10

        # 检查换行符到当前位置之间是否只有空白字符
        between = before_text[last_newline + 1:position]
        return len(between.strip()) == 0

    def _locate_question_numbers_with_ocr(
        self,
        question_numbers: List[Dict[str, Any]],
        bbox: List[int],
        original_image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        使用OCR精确定位题号在图像中的坐标

        基于PaddleOCR返回的JSON格式解析:
        - rec_texts: 识别的文本列表
        - rec_scores: 对应的置信度分数
        - rec_boxes: 文本框坐标 [x1, y1, x2, y2]
        - rec_polys: 文本框四点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            [{'number': int, 'bbox': [x1,y1,x2,y2], 'confidence': float}]
        """
        # 裁剪文本块区域
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cropped_image = original_image[y1:y2, x1:x2]

        # 保存临时图像
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, cropped_image)

        try:
            # 执行OCR
            ocr_results = self.ocr_model.predict(input=tmp_path, return_word_box=True)

            if not ocr_results:
                raise ValueError("OCR未返回结果")

            # 从第一个结果中提取数据
            result = ocr_results[0]
            # 解析OCR结果 - 使用rec_texts, rec_scores, rec_polys
            rec_texts = result.get("rec_texts")
            rec_scores = result.get("rec_scores")
            rec_polys = result.get("rec_polys")

            if not rec_texts or len(rec_texts) == 0:
                raise ValueError("OCR未识别到文本")

            # 构建OCR行列表: [(text, score, bbox_points)]
            ocr_lines = []
            for i, text in enumerate(rec_texts):
                score = rec_scores[i] if i < len(rec_scores) else 0.0
                poly = rec_polys[i] if i < len(rec_polys) else None
                if poly is not None and poly.any():
                    ocr_lines.append((text, score, poly))

        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not ocr_lines:
            raise ValueError("OCR未返回有效结果")

        # 匹配题号
        positions = []

        for qn_info in question_numbers:
            target_number = str(qn_info['number'])

            # 在OCR结果中查找包含该题号的文本行
            best_match = None
            best_score = 0
            best_poly = None

            for line_text, line_conf, line_poly in ocr_lines:
                # 检查是否包含题号（支持"2."、"2、"等格式）
                if target_number in line_text:
                    # 优先选择置信度高的匹配
                    if line_conf > best_score:
                        best_score = line_conf
                        best_match = line_text
                        best_poly = line_poly

            if best_poly is not None and best_poly.any():
                # 将相对坐标转换为原图坐标
                original_bbox = self._convert_to_original_bbox(best_poly, bbox)

                positions.append({
                    'number': qn_info['number'],
                    'bbox': original_bbox,
                    'confidence': best_score
                })
                logger.debug(f"找到题号{qn_info['number']}: {best_match}, 置信度: {best_score:.3f}")
            else:
                logger.warning(f"OCR未找到题号{qn_info['number']}，将使用估算位置")

        return positions

    def _estimate_positions_from_text(
        self,
        question_numbers: List[Dict[str, Any]],
        bbox: List[int],
        content: str
    ) -> List[Dict[str, Any]]:
        """
        从文本位置估算题号的图像坐标（备用方案）

        基于题号在文本中的字符位置比例估算Y坐标
        """
        x1, y1, x2, y2 = bbox
        block_height = y2 - y1
        text_length = len(content)

        positions = []

        for qn_info in question_numbers:
            # 计算题号在文本中的相对位置
            relative_pos = qn_info['position'] / text_length if text_length > 0 else 0

            # 估算Y坐标
            estimated_y = y1 + int(block_height * relative_pos)

            positions.append({
                'number': qn_info['number'],
                'bbox': [x1, estimated_y, x2, estimated_y + 30],  # 假设题号高度30像素
                'confidence': 0.5,  # 估算的置信度较低
                'estimated': True
            })

        return positions

    def _convert_to_original_bbox(
        self,
        relative_bbox: List[List[float]],
        base_bbox: List[int]
    ) -> List[int]:
        """
        将OCR的相对坐标转换为原图坐标

        Args:
            relative_bbox: OCR返回的四点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            base_bbox: 裁剪区域在原图中的坐标 [x1, y1, x2, y2]

        Returns:
            原图坐标 [x1, y1, x2, y2]
        """
        xs = [point[0] for point in relative_bbox]
        ys = [point[1] for point in relative_bbox]

        rel_x1, rel_y1 = min(xs), min(ys)
        rel_x2, rel_y2 = max(xs), max(ys)

        orig_x1 = int(base_bbox[0] + rel_x1)
        orig_y1 = int(base_bbox[1] + rel_y1)
        orig_x2 = int(base_bbox[0] + rel_x2)
        orig_y2 = int(base_bbox[1] + rel_y2)

        return [orig_x1, orig_y1, orig_x2, orig_y2]

    def _split_by_positions(
        self,
        text_block: Dict[str, Any],
        question_positions: List[Dict[str, Any]],
        question_numbers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        根据题号位置分割文本块

        策略：按题号的Y坐标（垂直位置）划分边界
        """
        split_questions = []
        content = text_block.get('block_content', '')
        base_bbox = text_block.get('block_bbox', [])

        # 按Y坐标排序题号位置
        sorted_positions = sorted(question_positions, key=lambda x: x['bbox'][1])

        for i, qn_pos in enumerate(sorted_positions):
            current_qn = next(
                (q for q in question_numbers if q['number'] == qn_pos['number']),
                None
            )

            if not current_qn:
                continue

            # 提取题目内容
            if i < len(sorted_positions) - 1:
                next_qn = next(
                    (q for q in question_numbers if q['number'] == sorted_positions[i+1]['number']),
                    None
                )
                if next_qn:
                    question_content = content[current_qn['position']:next_qn['position']].strip()
                else:
                    question_content = content[current_qn['position']:].strip()
            else:
                question_content = content[current_qn['position']:].strip()

            # 计算题目的bbox
            top = qn_pos['bbox'][1]

            if i < len(sorted_positions) - 1:
                bottom = sorted_positions[i + 1]['bbox'][1]
            else:
                bottom = base_bbox[3]

            # 确保高度合理
            if bottom - top < self.config.min_question_height:
                bottom = top + self.config.min_question_height

            question_bbox = [
                base_bbox[0],  # left
                top,           # top
                base_bbox[2],  # right
                bottom         # bottom
            ]

            split_questions.append({
                'block_label': 'text',
                'block_content': question_content,
                'block_bbox': question_bbox,
                'question_number': current_qn['number'],
                'original_block_id': text_block.get('block_id'),
                'split_from_merged': True,
                'confidence': qn_pos.get('confidence', 1.0)
            })

        return split_questions
