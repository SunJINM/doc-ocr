"""
试卷结构化分析测试脚本 - PaddleOCR VL 版本
使用 PaddleOCR VL API 进行版面检测和文本识别

流程：
1. PaddleOCR VL API 检测+识别（替代 PaddleOCR v5）
2. 绘制 ID 标记图
3. Qwen-VL 语义聚合
4. 后处理合并
"""

import os
import sys
import cv2
import copy
import json
import base64
import logging
import requests
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

# 添加 backend 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from openai import OpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 数据结构定义 ====================

@dataclass
class DetectionBlock:
    """检测块（PaddleOCR VL API 返回）"""
    id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    text: str
    label: str = "text"  # doc_title, paragraph_title, text, image, aside_text, number等
    block_order: Optional[int] = None
    group_id: Optional[int] = None


@dataclass
class QuestionGroup:
    """题目分组"""
    type: str  # doc_title, header, question, aside
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


# ==================== 第一步：PaddleOCR VL API 检测+识别 ====================

class QwenVLOCRRecognizer:
    """Qwen-VL OCR 识别器（用于二次识别文本块）"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-f436e171e65c4999bb7e8203f0862317")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        if not self.api_key:
            logger.warning("未设置 DASHSCOPE_API_KEY，Qwen-VL OCR 功能将不可用")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def recognize_image(self, image_data: bytes) -> str:
        """
        使用 qwen-vl-ocr-latest 模型识别图片文本
        
        Args:
            image_data: 图片二进制数据
            
        Returns:
            识别的文本内容
        """
        if not self.api_key:
            logger.warning("未设置 DASHSCOPE_API_KEY，无法调用 Qwen-VL OCR")
            return ""
        
        # 将图片数据编码为 base64
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
        
        # 初始化 Qwen-VL OCR 识别器（用于文本块的二次识别）
        if enable_ocr_fallback:
            self.ocr_recognizer = QwenVLOCRRecognizer(api_key=ocr_api_key)
        else:
            self.ocr_recognizer = None
    
    def detect_and_recognize(self, image_path: str) -> List[DetectionBlock]:
        """
        使用 PaddleOCR VL API 进行检测和识别
        
        Args:
            image_path: 图片路径
            
        Returns:
            检测块列表
        """
        # 读取图片并编码为 base64
        with open(image_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("ascii")
        
        headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "file": file_data,
            "fileType": 1,  # 1=图像, 0=PDF
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
            
            # 解析返回的检测结果
            for res in result.get("layoutParsingResults", []):
                pruned = res.get("prunedResult", {})
                parsing_list = pruned.get("parsing_res_list", [])
                
                for item in parsing_list:
                    bbox = item.get("block_bbox", [0, 0, 0, 0])
                    
                    # 确保 bbox 格式正确
                    if isinstance(bbox, list) and len(bbox) == 4:
                        bbox = [int(b) for b in bbox]
                    else:
                        logger.warning(f"跳过无效 bbox: {bbox}")
                        continue
                    
                    blocks.append(DetectionBlock(
                        id=item.get("block_id", len(blocks)),
                        bbox=bbox,
                        text=item.get("block_content", "").replace("\n", " "),
                        label=item.get("block_label", "text"),
                        block_order=item.get("block_order"),
                        group_id=item.get("group_id")
                    ))
            
            logger.info(f"PaddleOCR VL 检测完成，有效区域: {len(blocks)}")
            
            # 对文本块进行二次识别
            if self.enable_ocr_fallback:
                blocks = self._process_empty_text_blocks(image_path, blocks)
            
            # 打印前3个结果示例
            for i, block in enumerate(blocks[:3]):
                logger.info(
                    f"  块{i}: label={block.label}, "
                    f"text='{block.text[:30]}...', "
                    f"bbox={block.bbox}"
                )
            
            return blocks
            
        except Exception as e:
            logger.error(f"PaddleOCR VL API 调用失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def _process_empty_text_blocks(self, image_path: str, blocks: List[DetectionBlock]) -> List[DetectionBlock]:
        """
        处理文本块：裁剪图片并使用 Qwen-VL OCR 识别
        
        Args:
            image_path: 原始图片路径
            blocks: 检测块列表
            
        Returns:
            更新后的检测块列表
        """
        if not self.ocr_recognizer:
            return blocks
        
        # 读取原始图片
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"无法读取图片: {image_path}")
            return blocks
        
        # 找出所有文本类型但内容为空的块
        # VL模型有问题
        empty_text_blocks = [
            b for b in blocks
            # if b.label == "text" and not b.text.strip()
            if b.label == "text"
        ]
        
        if not empty_text_blocks:
            logger.info("没有发现文本块")
            return blocks
        
        logger.info(f"发现 {len(empty_text_blocks)} 个文本块，开始二次识别...")
        
        recognized_count = 0
        for block in empty_text_blocks:
            x1, y1, x2, y2 = block.bbox
            
            # 裁剪子图片
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                logger.warning(f"块 {block.id} 裁剪失败，跳过")
                continue
            
            # 编码为字节
            _, buffer = cv2.imencode('.jpg', cropped)
            image_bytes = buffer.tobytes()
            
            # 使用 Qwen-VL OCR 识别
            logger.info(f"  识别块 {block.id} (bbox={block.bbox})...")
            recognized_text = self.ocr_recognizer.recognize_image(image_bytes)
            
            if recognized_text:
                block.text = recognized_text.replace("\n", " ")
                recognized_count += 1
                logger.info(f"    ✓ 识别成功: '{recognized_text[:50]}...'")
            else:
                logger.info(f"    ✗ 识别失败或无文本")
            import time
            time.sleep(0.5)
        
        logger.info(f"二次识别完成: {recognized_count}/{len(empty_text_blocks)} 个块成功识别")
        
        return blocks


# ==================== 第二步：绘制 ID 标记图 ====================

class VisualMarker:
    """视觉标记绘制器 (Set-of-Mark)"""
    
    # 不同标签的颜色映射
    LABEL_COLORS = {
        "doc_title": (255, 0, 0),       # 蓝色
        "paragraph_title": (0, 165, 255), # 橙色
        "text": (0, 0, 255),             # 红色
        "image": (0, 255, 0),            # 绿色
        "aside_text": (128, 128, 128),   # 灰色
        "number": (255, 255, 0),         # 青色
        "inline_formula": (255, 0, 255), # 紫色
    }
    
    @staticmethod
    def draw_marks(image_path: str, blocks: List[DetectionBlock], output_path: str) -> str:
        """
        在图片上绘制 ID 标记
        
        Args:
            image_path: 原始图片路径
            blocks: 检测块列表
            output_path: 输出图片路径
            
        Returns:
            标记后的图片路径
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        marked_img = copy.deepcopy(image)
        
        for block in blocks:
            x1, y1, x2, y2 = block.bbox
            
            # 根据 label 选择颜色
            color = VisualMarker.LABEL_COLORS.get(block.label, (0, 0, 255))
            
            # 1. 画边框
            cv2.rectangle(marked_img, (x1, y1), (x2, y2), color, 2)
            
            # 2. 画 ID 标签（黄色背景 + 黑色文字）
            label = str(block.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 标签背景
            cv2.rectangle(
                marked_img,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                (0, 255, 255),  # 黄色背景
                -1
            )
            
            # 标签文字
            cv2.putText(
                marked_img,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),  # 黑色文字
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


# ==================== 第三步：Qwen-VL 语义聚合 ====================

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
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "14ebfc74-500c-46d5-a58b-61ac61341018")
        self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3"
        
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
            
            # 解析 JSON
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            groups = json.loads(result_text)
            
            return groups
            
        except Exception as e:
            logger.error(f"Doubao-VL 调用失败: {e}")
            raise

# ==================== 第四步：后处理合并 ====================

class PostProcessor:
    """后处理器"""
    
    @staticmethod
    def split_multi_question_text(text: str) -> List[Dict]:
        """
        检测并拆分包含多道题目的文本
        
        Args:
            text: 文本内容
            
        Returns:
            拆分后的题目列表，每个元素包含 question_number 和 start_pos
        """
        import re
        
        # 匹配题号的正则表达式
        # 匹配: "1." "2." "3." 或 "1、" "2、" 或 "第1题" "第2题" 等
        patterns = [
            r'^\s*(\d+)\.\s+',  # 1. 2. 3.
            r'^\s*(\d+)、\s+',  # 1、2、3、
            r'^\s*第(\d+)题',   # 第1题 第2题
            r'\n\s*(\d+)\.\s+', # 换行后的 1. 2. 3.
            r'\n\s*(\d+)、\s+', # 换行后的 1、2、3、
            r'\n\s*第(\d+)题',  # 换行后的 第1题 第2题
        ]
        
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                question_num = int(match.group(1))
                start_pos = match.start()
                matches.append({
                    'question_number': question_num,
                    'start_pos': start_pos,
                    'match_text': match.group(0)
                })
        
        # 按位置排序并去重
        matches.sort(key=lambda x: x['start_pos'])
        
        # 去重：如果同一位置有多个匹配，保留第一个
        unique_matches = []
        last_pos = -1
        for match in matches:
            if match['start_pos'] != last_pos:
                unique_matches.append(match)
                last_pos = match['start_pos']
        
        return unique_matches
    
    @staticmethod
    def merge_bboxes(blocks: List[DetectionBlock], group_ids: List[int]) -> Dict:
        """
        合并多个块的边界框
        
        Args:
            blocks: 所有检测块
            group_ids: 属于同一组的块 ID 列表
            
        Returns:
            合并后的边界框和文本
        """
        if not group_ids:
            return {"bbox": [0, 0, 0, 0], "text": ""}
        
        # 创建 ID 到块的映射
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
        
        return {
            "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
            "text": " ".join(texts)
        }
    
    @staticmethod
    def validate_groups(blocks: List[DetectionBlock], groups: List[Dict]) -> List[Dict]:
        """
        验证分组结果，确保所有 ID 都被分配
        
        Args:
            blocks: 所有检测块
            groups: VL 返回的分组
            
        Returns:
            验证后的分组
        """
        all_ids = {b.id for b in blocks}
        assigned_ids = set()
        
        for group in groups:
            assigned_ids.update(group.get('block_ids', []))
        
        # 找出未分配的 ID
        unassigned = all_ids - assigned_ids
        
        if unassigned:
            logger.warning(f"发现未分配的 ID: {unassigned}")
            # 兜底策略：将未分配的 ID 各自作为独立组
            for uid in unassigned:
                groups.append({
                    "type": "unknown",
                    "block_ids": [uid]
                })
        
        return groups
    
    @staticmethod
    def process(blocks: List[DetectionBlock], groups: List[Dict]) -> List[QuestionGroup]:
        """
        执行后处理
        
        Args:
            blocks: 所有检测块
            groups: VL 返回的分组
            
        Returns:
            处理后的题目分组列表
        """
        # 验证分组
        validated_groups = PostProcessor.validate_groups(blocks, groups)
        
        # 创建 ID 到块的映射
        block_map = {b.id: b for b in blocks}
        
        result = []
        for group in validated_groups:
            group_type = group.get('type', 'unknown')
            block_ids = group.get('block_ids', [])
            
            # 只对 question 类型进行多题拆分检测
            if group_type == 'question' and len(block_ids) > 0:
                # 检查是否包含多道题目
                merged_text = " ".join([block_map[bid].text for bid in block_ids if bid in block_map])
                question_matches = PostProcessor.split_multi_question_text(merged_text)
                
                # 如果检测到多道题目（至少2道连续题目）
                if len(question_matches) >= 2:
                    # 检查题号是否连续
                    question_numbers = [m['question_number'] for m in question_matches]
                    is_consecutive = all(
                        question_numbers[i+1] == question_numbers[i] + 1
                        for i in range(len(question_numbers) - 1)
                    )
                    
                    if is_consecutive:
                        logger.warning(
                            f"检测到合并的多道题目 (题号 {question_numbers[0]}-{question_numbers[-1]})，"
                            f"block_ids={block_ids}。建议检查 VL 分组结果。"
                        )
                        logger.info(f"  合并文本预览: {merged_text[:150]}...")
                        
                        # 目前策略：保持原样但记录警告
                        # 未来可以实现更细粒度的拆分
            
            merged = PostProcessor.merge_bboxes(blocks, block_ids)
            result.append(QuestionGroup(
                type=group_type,
                block_ids=block_ids,
                merged_bbox=merged['bbox'],
                merged_text=merged['text']
            ))
        
        return result


# ==================== 主流程编排 ====================

class ExamPaperAnalyzerVL:
    """试卷分析器 - PaddleOCR VL 版本"""
    
    def __init__(
        self,
        api_url: str = None,
        token: str = None,
        api_key: str = None,
        vl_model: str = "qwen-vl-max",
        enable_ocr_fallback: bool = True,
        vl_type: str = "doubao"
    ):
        self.vl_model = vl_model
        
        # 初始化各组件
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
        
        self.visual_marker = VisualMarker()
        if vl_type == "doubao":
            self.vl_aggregator = DoubaoVLAggregator()
        else:
            self.vl_aggregator = QwenVLAggregator(api_key=api_key)
        self.post_processor = PostProcessor()
    
    def analyze(
        self,
        image_path: str,
        output_dir: str = "./output",
        confidence_threshold: float = 0.5
    ) -> AnalysisResult:
        """
        执行完整的试卷分析流程
        
        Args:
            image_path: 试卷图片路径
            output_dir: 输出目录
            confidence_threshold: 置信度阈值（VL API 不使用此参数，保留接口兼容性）
            
        Returns:
            分析结果
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        image_name = Path(image_path).stem
        
        logger.info("=" * 50)
        logger.info(f"开始分析试卷 (VL版): {image_path}")
        logger.info("=" * 50)
        
        # Step 1: PaddleOCR VL API 检测+识别
        logger.info("\n[Step 1] PaddleOCR VL API 检测+识别...")
        blocks = self.detector.detect_and_recognize(image_path)
        
        # 获取图片尺寸
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # 立即保存 PaddleOCR 原始数据（在第二步之前）
        paddle_raw_json = {
            "image_size": {"width": width, "height": height},
            "blocks": [
                {
                    "id": b.id,
                    "bbox": b.bbox,
                    "text": b.text,
                    "label": b.label,
                    "block_order": b.block_order,
                    "group_id": b.group_id
                }
                for b in blocks
            ]
        }
        
        paddle_raw_path = os.path.join(output_dir, f"{image_name}_vl_paddle_raw.json")
        with open(paddle_raw_path, 'w', encoding='utf-8') as f:
            json.dump(paddle_raw_json, f, ensure_ascii=False, indent=2)
        
        logger.info(f"PaddleOCR 原始数据已保存: {paddle_raw_path}")
        
        # Step 2: 绘制标记图
        logger.info("\n[Step 2] 绘制 ID 标记图...")
        marked_image_path = os.path.join(output_dir, f"{image_name}_vl_marked.jpg")
        self.visual_marker.draw_marks(image_path, blocks, marked_image_path)
        
        # Step 3: VL 语义聚合
        logger.info("\n[Step 3] VL 语义聚合...")
        groups = self.vl_aggregator.aggregate(marked_image_path, blocks, self.vl_model)
        
        # Step 4: 后处理
        logger.info("\n[Step 4] 后处理合并...")
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
        # 保存完整结果（包含 VL 聚合后的分组）
        output_json = {
            "image_size": result.image_size,
            "marked_image": result.marked_image_path,
            "blocks": [
                {
                    "id": b.id,
                    "bbox": b.bbox,
                    "text": b.text,
                    "label": b.label,
                    "block_order": b.block_order,
                    "group_id": b.group_id
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
        
        json_path = os.path.join(output_dir, f"{image_name}_vl_result.json")
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
    # 配置
    IMAGE_PATH = r"D:\WorkProjects\doc-ocr\input\mifeng_doubao_1.jpg"
    # IMAGE_PATH = r"D:\software\downloads\package\d67393df-2739-42ab-abe2-afa281e7e350.png"
    OUTPUT_DIR = "./output/exam_analysis_vl"
    VL_MODEL = "ep-20251025164648-d66ns"  # 或 "qwen-vl-plus"
    
    # 检查图片是否存在
    if not os.path.exists(IMAGE_PATH):
        logger.error(f"图片不存在: {IMAGE_PATH}")
        logger.info("请修改 IMAGE_PATH 变量为你的试卷图片路径")
        return
    
    # 创建分析器
    analyzer = ExamPaperAnalyzerVL(vl_model=VL_MODEL, vl_type="qwen")
    
    # 执行分析
    result = analyzer.analyze(
        image_path=IMAGE_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)
    print(f"检测到区域数量: {len(result.blocks)}")
    print(f"聚合后分组数量: {len(result.groups)}")
    print(f"标记图片路径: {result.marked_image_path}")
    print("\n分组详情:")
    for idx, group in enumerate(result.groups):
        print(f"  [{idx}] {group.type}: IDs={group.block_ids}, bbox={group.merged_bbox}")
        print(f"       文本: {group.merged_text[:100]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()