"""
试卷结构化分析测试脚本 - V2 版本（增强拆分）
基于 test_exam_paper_analysis_vl_ocr.py，增加 OCR 精确定位 + 混合策略拆分

新增功能：
1. OCR 行级坐标精确拆分（参考 question_splitter.py）
2. 规则过滤误匹配
3. LLM 辅助判断（可选）

流程：
1. PaddleOCR VL API 检测+识别
2. ✨ OCR 精确拆分合并的题目（新增）
3. 绘制 ID 标记图
4. VL 语义聚合
5. 后处理合并
"""

import os
import re
import sys
import cv2
import copy
import json
import base64
import logging
import requests
import tempfile
import numpy as np
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

# 添加 backend 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from paddleocr import PaddleOCR

# 配置日志（强制设置，防止被 PaddleOCR 覆盖）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Python 3.8+ 强制重新配置
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 确保根日志器也是 INFO 级别
logging.getLogger().setLevel(logging.INFO)


# ==================== 数据结构定义 ====================

@dataclass
class DetectionBlock:
    """检测块"""
    id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    text: str
    label: str = "text"
    block_order: Optional[int] = None
    group_id: Optional[int] = None
    question_number: Optional[int] = None  # 新增：题号
    split_from_merged: bool = False  # 新增：是否由拆分产生


@dataclass
class QuestionGroup:
    """题目分组"""
    type: str
    block_ids: List[int]
    merged_bbox: Optional[List[int]] = None
    merged_text: str = ""


@dataclass
class AnalysisResult:
    """分析结果"""
    blocks: List[DetectionBlock] = field(default_factory=list)
    groups: List[QuestionGroup] = field(default_factory=list)
    marked_image_path: str = ""
    image_size: Dict[str, int] = field(default_factory=dict)


# ==================== 新增：OCR 精确拆分器 ====================

class OCRBasedSplitter:
    """基于 OCR 行级坐标的精确拆分器"""

    # 题号正则模式
    QUESTION_PATTERNS = [
        r'^(\d+)[\.、]\s*',     
        r'^\((\d+)\)\s*',          
        r'^第(\d+)题\s*',         
        r'^\[(\d+)\]\s*',         
        r'^[【](\d+)[】]\s*',      
    ]

    def __init__(self, ocr_model):
        """
        Args:
            ocr_model: PaddleOCR 实例（支持 return_word_box=True）
        """
        self.ocr_model = ocr_model

    def split(self, block: DetectionBlock, original_image: np.ndarray) -> List[DetectionBlock]:
        """拆分单个文本块"""
        import re

        # 检测题号
        matches = []
        for pattern in self.QUESTION_PATTERNS:
            for match in re.finditer(pattern, block.text, re.MULTILINE):
                number = int(match.group(1))
                position = match.start()
                matches.append({
                    'question_number': number,
                    'start_pos': position,
                    'matched_str': match.group(0)
                })

        # 按位置排序并去重
        matches.sort(key=lambda x: x['start_pos'])
        unique_matches = []
        last_pos = -10
        for m in matches:
            if m['start_pos'] - last_pos > 5:
                unique_matches.append(m)
                last_pos = m['start_pos']

        if len(unique_matches) < 2:
            return [block]

        logger.info(f"块 {block.id} 检测到 {len(unique_matches)} 个题号: {[m['question_number'] for m in unique_matches]}")

        # 使用 OCR 获取精确坐标
        try:
            sub_bboxes = self._get_precise_sub_bbox_with_ocr(
                block, unique_matches, original_image
            )
        except Exception as e:
            logger.warning(f"OCR 精确定位失败: {e}，使用估算")
            sub_bboxes = self._estimate_positions(block, unique_matches)

        # 创建子块
        sub_blocks = []
        for i, (match, bbox) in enumerate(zip(unique_matches, sub_bboxes)):
            start = match['start_pos']
            end = unique_matches[i+1]['start_pos'] if i+1 < len(unique_matches) else len(block.text)
            sub_text = block.text[start:end].strip()

            # 生成新的唯一整数 ID（避免字符串 ID）
            # 使用原 ID * 100 + 子块索引的方式
            new_id = block.id * 100 + i

            sub_blocks.append(DetectionBlock(
                id=new_id,
                bbox=bbox,
                text=sub_text,
                label=block.label,
                question_number=match['question_number'],
                split_from_merged=True
            ))

        logger.info(f"块 {block.id} 拆分完成: 1 → {len(sub_blocks)} 块")
        return sub_blocks

    def _get_precise_sub_bbox_with_ocr(
        self,
        block: DetectionBlock,
        question_matches: List[Dict],
        original_image: np.ndarray
    ) -> List[List[int]]:
        """核心：基于 OCR 行级坐标精确计算 bbox"""
        x1, y1, x2, y2 = block.bbox
        cropped_image = original_image[y1:y2, x1:x2]

        # 保存临时图像
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, cropped_image)

        try:
            # 关键：return_word_box=True 获取行级坐标
            ocr_results = self.ocr_model.predict(input=tmp_path)

            result = ocr_results[0]
            rec_texts = result.get("rec_texts")
            rec_scores = result.get("rec_scores")
            rec_polys = result.get("rec_polys")

            if not rec_texts:
                raise ValueError("OCR 未返回结果")

            # 构建 OCR 行列表
            ocr_lines = [
                (rec_texts[i], rec_scores[i], rec_polys[i])
                for i in range(len(rec_texts))
                if rec_polys[i] is not None and rec_polys[i].any()
            ]
        finally:
            os.remove(tmp_path)

        if not ocr_lines:
            raise ValueError("OCR 未返回有效结果")
        
        logger.info(f"OCR识别的行级信息为：{ocr_lines}")

        # 匹配题号到 OCR 行
        sub_bboxes = []
        for i, match in enumerate(question_matches):
            target_number = str(match['question_number'])

            # 查找包含题号的文本行
            question_line_bbox = None
            for line_text, line_conf, line_poly in ocr_lines:
                if self._contains_question_number(line_text, target_number):
                    question_line_bbox = self._convert_poly_to_bbox(line_poly, block.bbox)
                    logger.debug(f"  找到题号 {target_number} 在行: {line_text[:30]}")
                    break

            if not question_line_bbox:
                logger.warning(f"  OCR 未找到题号 {target_number}，使用估算")
                raise ValueError(f"未找到题号 {target_number}")

            # 计算当前题目的边界
            top = question_line_bbox[1]

            # 下一题的起始作为当前题的结束
            if i + 1 < len(question_matches):
                next_number = str(question_matches[i + 1]['question_number'])

                bottom = y2
                for line_text, _, line_poly in ocr_lines:
                    if self._contains_question_number(line_text, next_number):
                        next_bbox = self._convert_poly_to_bbox(line_poly, block.bbox)
                        bottom = next_bbox[1]
                        break
            else:
                bottom = y2

            sub_bboxes.append([x1, top, x2, bottom])

        return sub_bboxes

    def _contains_question_number(self, text: str, number: str) -> bool:
        """检查文本是否包含指定题号"""
        import re
        patterns = [
            rf'^{number}\.',
            rf'^{number}、',
            rf'第{number}题',
            rf'\({number}\)',
        ]
        for pattern in patterns:
            if re.search(pattern, text.strip()):
                return True
        return False

    def _convert_poly_to_bbox(
        self,
        poly: List[List[float]],
        base_bbox: List[int]
    ) -> List[int]:
        """四点坐标 → 矩形 bbox（原图坐标）"""
        xs = [point[0] for point in poly]
        ys = [point[1] for point in poly]

        abs_x1 = int(base_bbox[0] + min(xs))
        abs_y1 = int(base_bbox[1] + min(ys))
        abs_x2 = int(base_bbox[0] + max(xs))
        abs_y2 = int(base_bbox[1] + max(ys))

        return [abs_x1, abs_y1, abs_x2, abs_y2]

    def _estimate_positions(
        self,
        block: DetectionBlock,
        question_matches: List[Dict]
    ) -> List[List[int]]:
        """降级方案：线性估算"""
        x1, y1, x2, y2 = block.bbox
        block_height = y2 - y1
        text_length = len(block.text)

        positions = []
        for qn_info in question_matches:
            relative_pos = qn_info['start_pos'] / text_length if text_length > 0 else 0
            estimated_y = y1 + int(block_height * relative_pos)

            positions.append([x1, estimated_y, x2, estimated_y + 30])

        return positions


# ==================== 新增：规则过滤器 ====================

class ContextAwareSplitter:
    """基于规则的拆分器"""

    # 排除模式（不拆分）
    EXCLUDE_PATTERNS = [
        r'^(\d+)[\.、]\s*',        # 1. 或 1、
        r'^\((\d+)\)\s*',          # (1)
        r'^第(\d+)题\s*',          # 第1题
        r'^\[(\d+)\]\s*',          # [1]
        r'^[【](\d+)[】]\s*',       # 【1】
    ]

    def should_split(self, text: str) -> bool:
        """判断是否应该拆分"""
        question_numbers = self._detect_question_numbers(text)
        if len(question_numbers) <= 1:
            return False

        return True
    
    def _detect_question_numbers(self, text: str) -> List[Dict[str, Any]]:
        """
        检测文本中的所有题号

        Returns:
            [{'number': int, 'position': int, 'matched_str': str, 'type': str}]
        """
        question_number_patterns: List[str] = [
            r'^(\d+)[\.、]\s*',        # 1. 或 1、
            r'^\((\d+)\)\s*',          # (1)
            r'^第(\d+)题\s*',          # 第1题
            r'^\[(\d+)\]\s*',          # [1]
            r'^[【](\d+)[】]\s*',       # 【1】
        ]

        question_numbers = []

        for pattern in question_number_patterns:
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


# ==================== 原有组件（继承自 v1）====================

class QwenVLOCRRecognizer:
    """Qwen-VL OCR 识别器"""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-f436e171e65c4999bb7e8203f0862317")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not self.api_key:
            logger.warning("未设置 DASHSCOPE_API_KEY，Qwen-VL OCR 功能将不可用")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def recognize_image(self, image_data: bytes) -> str:
        """使用 qwen-vl-ocr-latest 模型识别图片文本"""
        if not self.api_key:
            logger.warning("未设置 DASHSCOPE_API_KEY，无法调用 Qwen-VL OCR")
            return ""

        base64_image = base64.b64encode(image_data).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model="qwen-vl-ocr-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            },
                            {
                                "type": "text",
                                "text": "请识别图片中的所有文字内容，直接返回识别结果，不要添加任何解释。"
                            }
                        ]
                    }
                ]
            )

            result_text = response.choices[0].message.content
            return result_text.strip()

        except Exception as e:
            logger.warning(f"Qwen-VL OCR 识别失败: {e}")
            return ""


class PaddleOCRVLDetector:
    """PaddleOCR VL API 检测器"""

    def __init__(self,
                 api_url: str = "https://l9h50fu036rbt0a5.aistudio-app.com/layout-parsing",
                 token: str = "fff80a7c88a38941601ff55e49aef457c1af6cbd",
                 enable_ocr_fallback: bool = True,
                 ocr_api_key: str = None):
        self.api_url = api_url
        self.token = token
        self.enable_ocr_fallback = enable_ocr_fallback

        if enable_ocr_fallback:
            self.ocr_recognizer = QwenVLOCRRecognizer(api_key=ocr_api_key)
        else:
            self.ocr_recognizer = None

    def detect_and_recognize(self, image_path: str) -> List[DetectionBlock]:
        """使用 PaddleOCR VL API 进行检测和识别"""
        with open(image_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("ascii")

        headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "file": file_data,
            "fileType": 1,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }

        logger.info(f"调用 PaddleOCR VL API: {image_path}")

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)

            if response.status_code != 200:
                raise Exception(f"API 调用失败: {response.status_code} - {response.text}")

            result = response.json()["result"]
            blocks = []

            for res in result.get("layoutParsingResults", []):
                pruned = res.get("prunedResult", {})
                parsing_list = pruned.get("parsing_res_list", [])

                for item in parsing_list:
                    bbox = item.get("block_bbox", [0, 0, 0, 0])

                    if isinstance(bbox, list) and len(bbox) == 4:
                        bbox = [int(b) for b in bbox]
                    else:
                        logger.warning(f"跳过无效 bbox: {bbox}")
                        continue

                    blocks.append(DetectionBlock(
                        id=item.get("block_id", len(blocks)),
                        bbox=bbox,
                        text=item.get("block_content", ""),
                        label=item.get("block_label", "text"),
                        block_order=item.get("block_order"),
                        group_id=item.get("group_id")
                    ))

            logger.info(f"PaddleOCR VL 检测完成，有效区域: {len(blocks)}")

            # 二次识别文本块
            if self.enable_ocr_fallback:
                blocks = self._process_empty_text_blocks(image_path, blocks)

            return blocks

        except Exception as e:
            logger.error(f"PaddleOCR VL API 调用失败: {e}")
            raise

    def _process_empty_text_blocks(self, image_path: str, blocks: List[DetectionBlock]) -> List[DetectionBlock]:
        """处理文本块：裁剪并识别"""
        if not self.ocr_recognizer:
            return blocks

        image = cv2.imread(image_path)
        if image is None:
            return blocks

        empty_text_blocks = [b for b in blocks if b.label == "text"]

        if not empty_text_blocks:
            return blocks

        logger.info(f"发现 {len(empty_text_blocks)} 个文本块，开始二次识别...")

        recognized_count = 0
        for block in empty_text_blocks:
            x1, y1, x2, y2 = block.bbox
            cropped = image[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            _, buffer = cv2.imencode('.jpg', cropped)
            image_bytes = buffer.tobytes()

            recognized_text = self.ocr_recognizer.recognize_image(image_bytes)

            if recognized_text:
                block.text = recognized_text
                recognized_count += 1

            import time
            time.sleep(0.5)

        logger.info(f"二次识别完成: {recognized_count}/{len(empty_text_blocks)} 个块成功识别")

        return blocks


# ==================== 其他组件（保持不变）====================

class VisualMarker:
    """视觉标记绘制器"""

    LABEL_COLORS = {
        "doc_title": (255, 0, 0),
        "paragraph_title": (0, 165, 255),
        "text": (0, 0, 255),
        "image": (0, 255, 0),
        "aside_text": (128, 128, 128),
        "number": (255, 255, 0),
    }

    @staticmethod
    def draw_marks(image_path: str, blocks: List[DetectionBlock], output_path: str) -> str:
        """在图片上绘制 ID 标记"""
        image = cv2.imread(image_path)
        marked_img = copy.deepcopy(image)

        for block in blocks:
            x1, y1, x2, y2 = block.bbox
            color = VisualMarker.LABEL_COLORS.get(block.label, (0, 0, 255))

            cv2.rectangle(marked_img, (x1, y1), (x2, y2), color, 2)

            label = str(block.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(
                marked_img,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                (0, 255, 255),
                -1
            )

            cv2.putText(
                marked_img,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

        cv2.imwrite(output_path, marked_img)
        logger.info(f"标记图片已保存: {output_path}")
        return output_path

    @staticmethod
    def draw_question_only_marks(
        image_path: str,
        blocks: List[DetectionBlock],
        groups: List[QuestionGroup],
        output_path: str
    ) -> str:
        """
        只绘制试题标注（过滤掉header、aside等）
        每道题目绘制一个整体大框
        
        Args:
            image_path: 原始图片路径
            blocks: 所有检测块
            groups: 分组结果
            output_path: 输出路径
        
        Returns:
            标注图片路径
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        marked_img = copy.deepcopy(image)
        
        # 只处理 type 为 "question" 的分组
        question_groups = [g for g in groups if g.type == "question"]
        
        # 为每道题目绘制整体边框
        for idx, group in enumerate(question_groups, start=1):
            if not group.merged_bbox:
                continue
            
            x1, y1, x2, y2 = group.merged_bbox
            
            # 1. 画红色边框（题目整体框）
            cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # 2. 画题号标签（黄色背景 + 黑色文字）
            label = f"Q{idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 标签背景
            cv2.rectangle(
                marked_img,
                (x1, y1 - text_h - 15),
                (x1 + text_w + 15, y1),
                (0, 255, 255),  # 黄色背景
                -1
            )
            
            # 标签文字
            cv2.putText(
                marked_img,
                label,
                (x1 + 7, y1 - 7),
                font,
                font_scale,
                (0, 0, 0),  # 黑色文字
                thickness
            )
        
        cv2.imwrite(output_path, marked_img)
        logger.info(f"试题标注图片已保存: {output_path} (共 {len(question_groups)} 道题目)")
        return output_path

class QwenVLAggregator:
    """Qwen-VL 语义聚合器"""
    
    # Prompt 模板
    PROMPT_TEMPLATE = """你是一个智能试卷结构化助手。

**任务**: 这是一个试卷页面，所有的内容块已经被框选并标记了数字 ID。同时我会提供每个 ID 对应的文字内容和类型标签。请根据试卷的**排版空间关系**和**语义逻辑**，将属于**同一道完整题目**的 ID 合并成一组。

**输入内容**:
```
{text_context}
```

**约束条件**:
1. 标签为 "doc_title" 的块作为文档标题，type 为 "doc_title"
2. 标签为 "paragraph_title" 的块如果是大题标题（如"一、选择题"），type 为 "header"
3. **【关键】每道题目必须单独成组，绝对不能将不同题号的题目合并！**
   - 通过题号识别不同题目（如"1."、"2."、"3."或"第1题"、"第2题"等）
   - 即使多道题目出现在同一个文本块中，也必须根据题号拆分成独立的 question 组
   - 一道完整题目可能包含：题干文本 + 选项 + 相关图片/表格
4. 图片/表格应归属到引用它的题目（通过"如图"、"如下表"等关键词判断）
5. 图片标题（如"第11题图"、"第12题图"）应归属到对应题号的题目，不要混淆
6. 标签为 "aside_text"、"number" 的块可以忽略或单独分组，type 为 "aside"
7. 请确保所有 ID 都被分配到某个组中

**特别注意**:
- 如果一个文本块包含多道题目（如"1. xxx 2. xxx 3. xxx"），该块只能分配给第一道题目
- 不同题号的题目绝对不能出现在同一个 question 组的 block_ids 中
- 每个 question 组应该只对应一道题目

**输出格式**:
请直接返回 JSON 格式，格式为列表，每个元素包含 `type` 和 `block_ids`。

**输出示例**:
```json
[
  {{"type": "doc_title", "block_ids": [6]}},
  {{"type": "header", "block_ids": [14]}},
  {{"type": "question", "block_ids": [16]}},
  {{"type": "question", "block_ids": [17, 18]}},
  {{"type": "question", "block_ids": [19]}},
  {{"type": "aside", "block_ids": [0, 1, 2, 3, 4, 5]}}
]
```

请分析图片并返回 JSON 结果："""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-f436e171e65c4999bb7e8203f0862317")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        if not self.api_key:
            logger.warning("未设置 DASHSCOPE_API_KEY，Qwen-VL 功能将不可用")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    @staticmethod
    def encode_image(image_path: str) -> str:
        """将图片编码为 Base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def build_text_context(self, blocks: List[DetectionBlock], max_text_len: int = 80) -> str:
        """
        构建文本上下文
        
        Args:
            blocks: 检测块列表
            max_text_len: 每个块的最大文本长度（用于节省 Token）
        """
        lines = []
        for block in blocks:
            text = block.text[:max_text_len] + "..." if len(block.text) > max_text_len else block.text
            if not text:
                text = "<空>"
            lines.append(f"ID {block.id} [{block.label}]: {text}")
        return "\n".join(lines)
    
    def aggregate(
        self,
        marked_image_path: str,
        blocks: List[DetectionBlock],
        model: str = "qwen-vl-max"
    ) -> List[Dict]:
        """
        使用 Qwen-VL 进行语义聚合
        
        Args:
            marked_image_path: 带标记的图片路径
            blocks: 检测块列表
            model: 使用的模型名称
            
        Returns:
            分组结果列表
        """
        if not self.api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY，无法调用 Qwen-VL")
        
        # 构建文本上下文
        text_context = self.build_text_context(blocks)
        prompt = self.PROMPT_TEMPLATE.format(text_context=text_context)
        
        # 编码图片
        base64_image = self.encode_image(marked_image_path)
        
        logger.info(f"调用 Qwen-VL 模型: {model}")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            result_text = response.choices[0].message.content
            logger.info(f"Qwen-VL 返回: {result_text[:200]}...")
            
            # 解析 JSON
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            groups = json.loads(result_text)
            
            return groups
            
        except Exception as e:
            logger.error(f"Qwen-VL 调用失败: {e}")
            raise


class DoubaoVLAggregator:
    """Doubao-VL 语义聚合器"""

    PROMPT_TEMPLATE = """你是一个智能试卷结构化助手。

**任务**: 请根据试卷的排版空间关系和语义逻辑，将属于同一道完整题目的 ID 合并成一组。

**输入内容**:
```
{text_context}
```

**约束条件**:
1. 标签为 "doc_title" 的块作为文档标题，type 为 "doc_title"
2. 标签为 "paragraph_title" 的块如果是大题标题（如"一、选择题"），type 为 "header"
3. **【关键】每道题目必须单独成组，绝对不能将不同题号的题目合并！**
4. 图片/表格应归属到引用它的题目
5. 图片标题（如"第11题图"）应归属到对应题号的题目
6. 标签为 "aside_text"、"number" 的块可以忽略或单独分组，type 为 "aside"

**输出格式**: 请直接返回 JSON 格式
```json
[
  {{"type": "doc_title", "block_ids": [6]}},
  {{"type": "header", "block_ids": [14]}},
  {{"type": "question", "block_ids": [16]}},
  {{"type": "aside", "block_ids": [0, 1, 2]}}
]
```

请分析图片并返回 JSON 结果："""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "14ebfc74-500c-46d5-a58b-61ac61341018")
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def build_text_context(self, blocks: List[DetectionBlock], max_text_len: int = 80) -> str:
        lines = []
        for block in blocks:
            text = block.text[:max_text_len] + "..." if len(block.text) > max_text_len else block.text
            if not text:
                text = "<空>"
            lines.append(f"ID {block.id} [{block.label}]: {text}")
        return "\n".join(lines)

    def aggregate(self, marked_image_path: str, blocks: List[DetectionBlock], model: str = "ep-20251025164648-d66ns") -> List[Dict]:
        text_context = self.build_text_context(blocks)
        prompt = self.PROMPT_TEMPLATE.format(text_context=text_context)

        base64_image = self.encode_image(marked_image_path)

        logger.info(f"调用 Doubao-VL 模型: {model}")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            result_text = response.choices[0].message.content
            logger.info(f"Doubao-VL 返回: {result_text[:200]}...")

            result_text = result_text.replace("```json", "").replace("```", "").strip()
            groups = json.loads(result_text)

            return groups

        except Exception as e:
            logger.error(f"Doubao-VL 调用失败: {e}")
            raise


class PostProcessor:
    """后处理器"""

    @staticmethod
    def merge_bboxes(blocks: List[DetectionBlock], group_ids: List[int]) -> Dict:
        if not group_ids:
            return {"bbox": [0, 0, 0, 0], "text": ""}

        block_map = {b.id: b for b in blocks}

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        texts = []

        for idx in group_ids:
            if idx not in block_map:
                continue
            block = block_map[idx]
            x1, y1, x2, y2 = block.bbox

            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

            if block.text:
                texts.append(block.text)

        # 处理空组（所有 ID 都不在 block_map 中）
        if min_x == float('inf'):
            logger.warning(f"组 {group_ids} 中所有 ID 都未找到对应 block，返回默认值")
            return {
                "bbox": [0, 0, 0, 0],
                "text": ""
            }

        return {
            "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
            "text": " ".join(texts)
        }

    @staticmethod
    def validate_groups(blocks: List[DetectionBlock], groups: List[Dict]) -> List[Dict]:
        all_ids = {b.id for b in blocks}
        assigned_ids = set()

        for group in groups:
            assigned_ids.update(group.get('block_ids', []))

        unassigned = all_ids - assigned_ids

        if unassigned:
            logger.warning(f"发现未分配的 ID: {unassigned}")
            for uid in unassigned:
                groups.append({
                    "type": "unknown",
                    "block_ids": [uid]
                })

        return groups

    @staticmethod
    def process(blocks: List[DetectionBlock], groups: List[Dict]) -> List[QuestionGroup]:
        validated_groups = PostProcessor.validate_groups(blocks, groups)

        result = []
        for group in validated_groups:
            group_type = group.get('type', 'unknown')
            block_ids = group.get('block_ids', [])

            merged = PostProcessor.merge_bboxes(blocks, block_ids)
            result.append(QuestionGroup(
                type=group_type,
                block_ids=block_ids,
                merged_bbox=merged['bbox'],
                merged_text=merged['text']
            ))

        return result


# ==================== 主流程编排（V2 增强版）====================

class ExamPaperAnalyzerVLV2:
    """试卷分析器 - V2 版本（增强拆分）"""

    def __init__(
        self,
        api_url: str = None,
        token: str = None,
        api_key: str = None,
        vl_model: str = "ep-20251025164648-d66ns",
        enable_ocr_fallback: bool = True,
        enable_ocr_split: bool = True,  # 新增：是否启用 OCR 精确拆分
        ocr_model_dir: str = None  # 新增：本地 OCR 模型路径
    ):
        self.vl_model = vl_model
        self.enable_ocr_split = enable_ocr_split

        # 初始化 PaddleOCR VL 检测器
        if api_url or token:
            self.detector = PaddleOCRVLDetector(
                api_url=api_url,
                token=token,
                enable_ocr_fallback=enable_ocr_fallback,
                ocr_api_key=api_key
            )
        else:
            self.detector = PaddleOCRVLDetector(
                enable_ocr_fallback=enable_ocr_fallback,
                ocr_api_key=api_key
            )

        # 新增：初始化本地 OCR（用于精确拆分）
        if enable_ocr_split:
            try:
                ocr_params = {
                    # 功能开关（对应界面“关闭”）
                    'use_doc_orientation_classify': False,
                    'use_doc_unwarping': False,
                    'use_textline_orientation': False,

                    # 文本检测相关（界面配置）
                    'text_det_thresh': 0.3,                  # 检测像素阈值
                    'text_det_box_thresh': 0.6,              # 检测框阈值
                    'text_det_unclip_ratio': 1.5,            # 扩张系数

                    # 文本识别相关（界面配置）
                    'text_rec_score_thresh': 0.0,            # 识别阈值
                    'return_word_box': False,                 # 返回单字框（界面勾选）

                    # 基础配置
                    'lang': 'ch',                            # 中文识别
                    'ocr_version': 'PP-OCRv5',                     # 若需指定OCR版本可填，如"PP-OCRv5"
                }
                if ocr_model_dir:
                    ocr_params['det_model_dir'] = os.path.join(ocr_model_dir, 'det')
                    ocr_params['rec_model_dir'] = os.path.join(ocr_model_dir, 'rec')
                    ocr_params['cls_model_dir'] = os.path.join(ocr_model_dir, 'cls')

                self.ocr_model = PaddleOCR(**ocr_params)
                self.ocr_splitter = OCRBasedSplitter(self.ocr_model)
                self.rule_filter = ContextAwareSplitter()
                logger.info("OCR 精确拆分器初始化成功")
            except Exception as e:
                logger.warning(f"OCR 精确拆分器初始化失败: {e}，将跳过拆分步骤")
                self.enable_ocr_split = False

        self.visual_marker = VisualMarker()
        self.vl_aggregator = QwenVLAggregator()
        self.post_processor = PostProcessor()

    def analyze(
        self,
        image_path: str,
        output_dir: str = "./output",
        confidence_threshold: float = 0.5
    ) -> AnalysisResult:
        """执行完整的试卷分析流程（V2 增强版）"""
        os.makedirs(output_dir, exist_ok=True)

        image_name = Path(image_path).stem

        logger.info("=" * 50)
        logger.info(f"开始分析试卷 (V2版): {image_path}")
        logger.info("=" * 50)

        # Step 1: PaddleOCR VL API 检测+识别
        logger.info("\n[Step 1] PaddleOCR VL API 检测+识别...")
        blocks = self.detector.detect_and_recognize(image_path)

        # 保存原始数据
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        paddle_raw_json = {
            "image_size": {"width": width, "height": height},
            "blocks": [
                {
                    "id": b.id,
                    "bbox": b.bbox,
                    "text": b.text,
                    "label": b.label,
                }
                for b in blocks
            ]
        }

        paddle_raw_path = os.path.join(output_dir, f"{image_name}_v2_paddle_raw.json")
        with open(paddle_raw_path, 'w', encoding='utf-8') as f:
            json.dump(paddle_raw_json, f, ensure_ascii=False, indent=2)

        logger.info(f"PaddleOCR 原始数据已保存: {paddle_raw_path}")

        # ✨ Step 2: OCR 精确拆分（新增）
        if self.enable_ocr_split:
            logger.info("\n[Step 2] ✨ OCR 精确拆分合并的题目...")
            original_image = cv2.imread(image_path)
            refined_blocks = []
            split_count = 0

            for block in blocks:
                if block.label == "text" and self.rule_filter.should_split(block.text):
                    logger.info(f"文本：{block.text} 需要拆分")
                    try:
                        sub_blocks = self.ocr_splitter.split(block, original_image)
                        if len(sub_blocks) > 1:
                            split_count += 1
                        refined_blocks.extend(sub_blocks)
                    except Exception as e:
                        logger.warning(f"块 {block.id} 拆分失败: {e}，保持原样")
                        refined_blocks.append(block)
                else:
                    refined_blocks.append(block)

            logger.info(f"拆分完成: {len(blocks)} → {len(refined_blocks)} 块（拆分了 {split_count} 个块）")
            blocks = refined_blocks

            # 保存拆分后的数据
            split_json = {
                "image_size": {"width": width, "height": height},
                "blocks": [
                    {
                        "id": b.id,
                        "bbox": b.bbox,
                        "text": b.text,
                        "label": b.label,
                        "question_number": b.question_number,
                        "split_from_merged": b.split_from_merged
                    }
                    for b in blocks
                ]
            }

            split_path = os.path.join(output_dir, f"{image_name}_v2_split.json")
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(split_json, f, ensure_ascii=False, indent=2)

            logger.info(f"拆分后数据已保存: {split_path}")
        else:
            logger.info("\n[Step 2] 跳过 OCR 精确拆分（未启用）")

        # Step 3: 绘制标记图
        logger.info("\n[Step 3] 绘制 ID 标记图...")
        marked_image_path = os.path.join(output_dir, f"{image_name}_v2_marked.jpg")
        self.visual_marker.draw_marks(image_path, blocks, marked_image_path)

        # Step 4: VL 语义聚合
        logger.info("\n[Step 4] VL 语义聚合...")
        groups = self.vl_aggregator.aggregate(marked_image_path, blocks, self.vl_model)

        # Step 5: 后处理
        logger.info("\n[Step 5] 后处理合并...")
        question_groups = self.post_processor.process(blocks, groups)

        result = AnalysisResult(
            blocks=blocks,
            groups=question_groups,
            marked_image_path=marked_image_path,
            image_size={"width": width, "height": height}
        )

        # 保存结果
        self._save_results(result, output_dir, image_name, image_path)

        logger.info("\n" + "=" * 50)
        logger.info("分析完成!")
        logger.info("=" * 50)

        return result

    def _save_results(self, result: AnalysisResult, output_dir: str, image_name: str, original_image_path: str = None):
        """保存分析结果"""
        output_json = {
            "image_size": result.image_size,
            "marked_image": result.marked_image_path,
            "blocks": [
                {
                    "id": b.id,
                    "bbox": b.bbox,
                    "text": b.text,
                    "label": b.label,
                    "question_number": b.question_number,
                    "split_from_merged": b.split_from_merged
                }
                for b in result.blocks
            ],
            "question_groups": [
                {
                    "type": g.type,
                    "block_ids": g.block_ids,
                    "merged_bbox": g.merged_bbox,
                    "merged_text": g.merged_text
                }
                for g in result.groups
            ]
        }

        json_path = os.path.join(output_dir, f"{image_name}_v2_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        logger.info(f"完整结果已保存: {json_path}")

        # 绘制只包含试题的标注图
        if original_image_path:
            question_only_path = os.path.join(output_dir, f"{image_name}_vl_questions_only.jpg")
            self.visual_marker.draw_question_only_marks(
                original_image_path,
                result.blocks,
                result.groups,
                question_only_path
            )
        
        # 绘制最终结果图
        self._draw_final_result(result, output_dir, image_name)

    def _draw_final_result(self, result: AnalysisResult, output_dir: str, image_name: str):
        """绘制最终分组结果图"""
        # 读取标记图
        image = cv2.imread(result.marked_image_path)
        
        # 为每个分组绘制不同颜色的边框
        colors = [
            (255, 0, 0),    # 蓝
            (0, 255, 0),    # 绿
            (255, 0, 255),  # 紫
            (0, 255, 255),  # 黄
            (255, 128, 0),  # 橙
            (128, 0, 255),  # 粉
        ]
        
        for idx, group in enumerate(result.groups):
            if group.merged_bbox:
                color = colors[idx % len(colors)]
                x1, y1, x2, y2 = group.merged_bbox
                
                # 绘制分组边框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # 绘制分组标签
                label = f"{group.type}_{idx}"
                cv2.putText(image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        output_path = os.path.join(output_dir, f"{image_name}_vl_final.jpg")
        cv2.imwrite(output_path, image)
        logger.info(f"最终结果图已保存: {output_path}")


# ==================== 测试入口 ====================

def main():
    """测试主函数"""
    IMAGE_PATH = r"D:\WorkProjects\doc-ocr\input\mifeng_doubao_1.jpg"
    OUTPUT_DIR = "./output/exam_analysis_vl_v2"
    VL_MODEL = "qwen-vl-max"

    if not os.path.exists(IMAGE_PATH):
        logger.error(f"图片不存在: {IMAGE_PATH}")
        return

    # 创建分析器（V2 版本，启用 OCR 精确拆分）
    analyzer = ExamPaperAnalyzerVLV2(
        vl_model=VL_MODEL,
        enable_ocr_split=True  # 启用 OCR 精确拆分
    )

    # 执行分析
    result = analyzer.analyze(
        image_path=IMAGE_PATH,
        output_dir=OUTPUT_DIR
    )

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("分析结果摘要（V2 版本）")
    print("=" * 60)
    print(f"检测到区域数量: {len(result.blocks)}")
    print(f"  - 拆分产生的子块: {sum(1 for b in result.blocks if b.split_from_merged)}")
    print(f"聚合后分组数量: {len(result.groups)}")
    print(f"标记图片路径: {result.marked_image_path}")
    print("\n分组详情:")
    for idx, group in enumerate(result.groups):
        print(f"  [{idx}] {group.type}: IDs={group.block_ids}")
        print(f"       文本: {group.merged_text[:100]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
