"""
题目提取主流程编排器

整合题目拆分和图文合并流程，提供完整的题目提取功能
"""
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import cv2

from .question_splitter import QuestionSplitter
from .question_merger import QuestionImageMerger

logger = logging.getLogger(__name__)


class ExamPaperQuestionExtractor:
    """试卷题目提取完整流程编排器"""

    def __init__(self, qwen_vl_client, ocr_model, config):
        """
        初始化题目提取器

        Args:
            qwen_vl_client: OpenAI SDK兼容的Qwen-VL客户端
            ocr_model: PaddleOCR实例
            config: ProcessingConfig配置实例
        """
        self.config = config
        self.splitter = QuestionSplitter(ocr_model, config)
        self.merger = QuestionImageMerger(qwen_vl_client, config)

    def extract_questions(
        self,
        result_data_path: str,
        original_image_path: str
    ) -> Dict[str, Any]:
        """
        完整的题目提取流程

        流程：
        1. 加载PP-OCR-VL检测结果
        2. 先拆分合并的题目块
        3. 再合并题目文本与配图（支持一题多图）
        4. 生成最终的结构化题目数据

        Args:
            result_data_path: PP-OCR-VL结果JSON路径
            original_image_path: 原始试卷图像路径

        Returns:
            结构化试卷数据
        """
        logger.info("=" * 60)
        logger.info("开始题目提取流程")
        logger.info("=" * 60)

        # 步骤1: 加载数据
        logger.info("步骤1: 加载检测结果和原始图像")
        with open(result_data_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        original_image = cv2.imread(original_image_path)

        if original_image is None:
            raise ValueError(f"无法加载图像: {original_image_path}")

        parsing_results = result_data['layoutParsingResults'][0]['prunedResult']
        layout_blocks = parsing_results['parsing_res_list']

        logger.info(f"加载完成：共{len(layout_blocks)}个布局块")

        # 步骤2: 分类处理
        logger.info("步骤2: 分类布局块")
        text_blocks = [b for b in layout_blocks if b.get('block_label') == 'text']
        image_blocks = [b for b in layout_blocks if b.get('block_label') == 'image']
        title_blocks = [b for b in layout_blocks
                       if b.get('block_label') in ['doc_title', 'paragraph_title']]

        logger.info(f"  文本块: {len(text_blocks)}")
        logger.info(f"  图像块: {len(image_blocks)}")
        logger.info(f"  标题块: {len(title_blocks)}")

        # 步骤3: 拆分合并的题目（重要：先拆分）
        logger.info("步骤3: 拆分合并的题目块")
        all_text_blocks = []

        for i, text_block in enumerate(text_blocks):
            logger.info(f"  处理文本块 {i+1}/{len(text_blocks)}")
            split_results = self.splitter.split_merged_questions(
                text_block,
                original_image
            )
            all_text_blocks.extend(split_results)

        logger.info(f"拆分完成：{len(text_blocks)} -> {len(all_text_blocks)}个文本块")

        # 步骤4: 合并题目与图像（支持一题多图）
        logger.info("步骤4: 合并题目文本与配图")
        merged_questions = self.merger.merge_text_and_images(
            all_text_blocks,
            image_blocks,
            original_image
        )

        logger.info(f"合并完成：{len(merged_questions)}个完整题目")

        # 步骤5: 构建最终结构
        logger.info("步骤5: 构建结构化数据")
        structured_exam = {
            'exam_info': self._extract_exam_info(title_blocks, layout_blocks),
            'questions': self._organize_questions(merged_questions),
            'metadata': {
                'total_questions': len(merged_questions),
                'with_images': len([q for q in merged_questions if q['has_image']]),
                'total_images': sum(len(q['images']) for q in merged_questions),
                'split_count': len([q for q in merged_questions
                                   if q.get('metadata', {}).get('split_from_merged')]),
                'processing_timestamp': datetime.now().isoformat(),
                'original_image': original_image_path,
                'result_data': result_data_path
            }
        }

        logger.info("=" * 60)
        logger.info("题目提取流程完成")
        logger.info(f"  总题目数: {structured_exam['metadata']['total_questions']}")
        logger.info(f"  带配图题目: {structured_exam['metadata']['with_images']}")
        logger.info(f"  总图片数: {structured_exam['metadata']['total_images']}")
        logger.info(f"  拆分题目数: {structured_exam['metadata']['split_count']}")
        logger.info("=" * 60)

        return structured_exam

    def _extract_exam_info(
        self,
        title_blocks: List[Dict[str, Any]],
        all_blocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        提取试卷基本信息

        从标题块中提取：试卷名称、时间、分数等
        """
        exam_info = {
            'title': '',
            'subject': '',
            'grade': '',
            'time_limit': '',
            'total_score': ''
        }

        # 提取标题
        for block in title_blocks:
            if block.get('block_label') == 'doc_title':
                title_text = block.get('block_content', '')
                exam_info['title'] = title_text

                # 尝试提取学科和年级
                if '数学' in title_text:
                    exam_info['subject'] = '数学'
                elif '语文' in title_text:
                    exam_info['subject'] = '语文'
                elif '英语' in title_text:
                    exam_info['subject'] = '英语'

                # 提取年级
                grade_match = re.search(r'[一二三四五六七八九]年级', title_text)
                if grade_match:
                    exam_info['grade'] = grade_match.group(0)

        # 提取时间和分数信息（通常在第一个text块）
        for block in all_blocks:
            if block.get('block_label') == 'text':
                content = block.get('block_content', '')

                # 提取时间
                time_match = re.search(r'时间[：:]\s*(\d+)\s*分钟', content)
                if time_match:
                    exam_info['time_limit'] = f"{time_match.group(1)}分钟"

                # 提取总分
                score_match = re.search(r'满分[：:]\s*([\d+]+)\s*分', content)
                if score_match:
                    exam_info['total_score'] = score_match.group(1)

                if exam_info['time_limit'] and exam_info['total_score']:
                    break

        return exam_info

    def _organize_questions(
        self,
        merged_questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        组织题目结构

        功能：
        1. 按题号排序
        2. 识别题目类型（选择、填空、解答等）
        3. 提取分数信息
        4. 格式化输出结构
        """
        organized = []

        # 按题号排序
        sorted_questions = sorted(
            merged_questions,
            key=lambda q: q.get('question_id') or 999
        )

        for question in sorted_questions:
            content = question.get('text_content', '')

            # 识别题目类型
            question_type = self._identify_question_type(content)

            # 提取分数
            score = self._extract_score(content)

            # 构建题目结构
            organized_question = {
                'id': question.get('question_id'),
                'type': question_type,
                'score': score,
                'content': {
                    'text': content,
                    'has_image': question.get('has_image', False),
                    'image_count': len(question.get('images', [])),
                    'images': []
                },
                'bbox': question.get('bbox', []),
                'metadata': question.get('metadata', {})
            }

            # 添加图像信息
            for img_info in question.get('images', []):
                organized_question['content']['images'].append({
                    'block_id': img_info.get('block_id'),
                    'bbox': img_info.get('block_bbox', []),
                    'vl_verified': 'vl_verification' in img_info,
                    'verification': img_info.get('vl_verification', {})
                })

            organized.append(organized_question)

        return organized

    def _identify_question_type(self, content: str) -> str:
        """
        识别题目类型

        基于题目内容特征判断类型
        """
        # 选择题：包含ABCD选项
        if re.search(r'[A-D][\.、]', content):
            return '选择题'

        # 填空题：包含括号或下划线
        if re.search(r'[（(]\s*[）)]|_{2,}', content):
            return '填空题'

        # 判断题：包含判断相关关键词
        if re.search(r'判断|对错|正确|错误|√|×|[（(]\s*[）)]', content) and len(content) < 200:
            return '判断题'

        # 计算题：包含计算相关关键词
        if re.search(r'计算|求值|求出|算出', content):
            return '计算题'

        # 解答题：包含解答相关关键词或内容较长
        if re.search(r'解答|证明|说明|分析|简述', content) or len(content) > 100:
            return '解答题'

        # 默认
        return '其他'

    def _extract_score(self, content: str) -> int:
        """
        提取题目分数

        从题目内容中提取分数信息
        """
        patterns = [
            r'[（(](\d+)分[）)]',
            r'(\d+)分',
            r'每题(\d+)分',
            r'共(\d+)分'
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue

        return 0
