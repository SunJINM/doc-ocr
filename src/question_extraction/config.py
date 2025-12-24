"""
配置管理模块
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class QwenVLConfig:
    """Qwen-VL视觉模型配置"""
    api_key: str = field(default_factory=lambda: os.environ.get('DASHSCOPE_API_KEY', 'sk-f436e171e65c4999bb7e8203f0862317'))
    base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    model: str = 'qwen-vl-plus'  # 或 'qwen-vl-max'
    temperature: float = 0.1  # 低温度提高稳定性
    max_tokens: int = 800
    timeout: int = 30


@dataclass
class OCRConfig:
    """PaddleOCR配置 - 新版API"""
    lang: str = 'ch'
    device: str = 'cpu'  # 'cpu' 或 'gpu'
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = True


@dataclass
class ProcessingConfig:
    """处理配置"""
    # 图文合并配置
    max_vertical_distance: int = 300  # 图文最大垂直距离（像素）
    max_horizontal_distance: int = 200  # 图文最大水平距离
    spatial_confidence_threshold: float = 0.85  # 空间置信度阈值
    vl_confidence_threshold: float = 0.7  # VL验证置信度阈值

    # 题目拆分配置
    question_number_patterns: List[str] = field(default_factory=lambda: [
        r'^(\d+)[\.、]\s*',        # 1. 或 1、
        r'^\((\d+)\)\s*',          # (1)
        r'^第(\d+)题\s*',          # 第1题
        r'^\[(\d+)\]\s*',          # [1]
        r'^[【](\d+)[】]\s*',       # 【1】
    ])
    min_question_height: int = 50  # 最小题目高度（像素）

    # 性能优化配置
    enable_cache: bool = True
    cache_dir: str = 'cache/vl_results'
    parallel_workers: int = 4
    enable_vl_batch: bool = True  # 批量VL验证
    vl_batch_size: int = 3

    # 一题多图配置
    max_images_per_question: int = 8  # 单题最多图片数
    image_clustering_enabled: bool = True  # 启用图像聚类
    same_row_threshold: int = 50  # 同一行的Y坐标差异阈值


# 全局配置实例
qwen_vl_config = QwenVLConfig()
ocr_config = OCRConfig()
processing_config = ProcessingConfig()
