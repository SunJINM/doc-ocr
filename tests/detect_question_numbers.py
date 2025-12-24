"""
数学试卷题目提取主程序

完整流程：
1. 先拆分合并的题目块
2. 再合并题目文本与配图（支持一题多图）
3. 生成结构化题目数据和可视化结果
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
from openai import OpenAI
from paddleocr import PaddleOCR

from question_extraction.config import (
    QwenVLConfig,
    OCRConfig,
    ProcessingConfig
)
from question_extraction.extractor import ExamPaperQuestionExtractor
from question_extraction.visualizer import ResultVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# Qwen-VL配置
qwen_config = QwenVLConfig()
logger.info("正在初始化模型...")
# 创建Qwen-VL客户端
qwen_client = OpenAI(
    api_key=qwen_config.api_key,
    base_url=qwen_config.base_url
)
# 创建OCR实例 - 新版API
ocr_config = OCRConfig()
ocr = PaddleOCR(
    lang=ocr_config.lang,
    device=ocr_config.device,
    use_doc_orientation_classify=ocr_config.use_doc_orientation_classify,
    use_doc_unwarping=ocr_config.use_doc_unwarping,
    use_textline_orientation=ocr_config.use_textline_orientation
)
# 创建处理配置
processing_config = ProcessingConfig()
# 3. 创建提取器
extractor = ExamPaperQuestionExtractor(
    qwen_client,
    ocr,
    processing_config
)
# 4. 执行提取
structured_exam = extractor._detect_question_numbers()
