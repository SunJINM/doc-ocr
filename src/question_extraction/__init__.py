"""
题目提取模块

包含题目拆分、图文合并、完整流程编排等功能
"""

from .config import QwenVLConfig, OCRConfig, ProcessingConfig
from .question_splitter import QuestionSplitter
from .question_merger import QuestionImageMerger
from .extractor import ExamPaperQuestionExtractor
from .visualizer import ResultVisualizer
from .evaluator import ExtractionEvaluator

__all__ = [
    'QwenVLConfig',
    'OCRConfig',
    'ProcessingConfig',
    'QuestionSplitter',
    'QuestionImageMerger',
    'ExamPaperQuestionExtractor',
    'ResultVisualizer',
    'ExtractionEvaluator'
]

__version__ = '1.0.0'
