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


class ContextAwareSplitter:
    """带语境感知的拆分器"""

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
                    logger.info(f"{pattern} 匹配到：{number}, {position}, {matched_str}")
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

if __name__ == "__main__":
    
    text = "3.右图用乘法算式表示是$(\\frac{2}{3}) \\times (\\frac{2}{5}) = (\\frac{4}{15})$。 4. $\\frac{5}{12}$时$=(\\boxed{25})$分 $\\frac{7}{20}$米$=(\\boxed{35})$厘米 $\\frac{4}{25}$吨$=(\\boxed{160})$千克 $\\frac{2}{5}m^3=(\\boxed{400})dm^3$"

    splitter = ContextAwareSplitter()
    res = splitter.should_split(text)
    print(res)
