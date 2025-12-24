"""
题目提取功能测试脚本

用于快速测试题目拆分和图文合并功能
"""
import os
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
from paddleocr import PaddleOCR

from question_extraction.config import ProcessingConfig, OCRConfig
from question_extraction.question_splitter import QuestionSplitter
from question_extraction.question_merger import QuestionImageMerger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_question_splitter():
    """测试题目拆分功能"""
    logger.info("=" * 60)
    logger.info("测试题目拆分功能")
    logger.info("=" * 60)

    # 初始化OCR
    ocr_config = OCRConfig()
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        device='cpu'
    )

    # 创建配置
    config = ProcessingConfig()

    # 创建拆分器
    splitter = QuestionSplitter(ocr, config)

    # 加载测试数据
    with open('output/result_data.json', 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    original_image = cv2.imread('data/shuxue/1.png')

    layout_blocks = result_data['layoutParsingResults'][0]['prunedResult']['parsing_res_list']
    text_blocks = [b for b in layout_blocks if b.get('block_label') == 'text']

    logger.info(f"找到 {len(text_blocks)} 个文本块")

    # 测试拆分
    all_split_blocks = []

    for i, text_block in enumerate(text_blocks):
        logger.info(f"\n处理文本块 {i+1}:")
        logger.info(f"  内容前50字: {text_block.get('block_content', '')[:50]}...")

        split_results = splitter.split_merged_questions(text_block, original_image)

        logger.info(f"  拆分结果: {len(split_results)} 个题目")

        for j, split_block in enumerate(split_results):
            logger.info(f"    题目 {j+1}:")
            logger.info(f"      题号: {split_block.get('question_number')}")
            logger.info(f"      坐标: {split_block.get('block_bbox')}")

        all_split_blocks.extend(split_results)

    logger.info(f"\n总计: {len(text_blocks)} 个文本块 -> {len(all_split_blocks)} 个独立题目")

    return all_split_blocks


def test_image_merger_mock():
    """测试图文合并功能（模拟VL客户端）"""
    logger.info("\n" + "=" * 60)
    logger.info("测试图文合并功能（模拟模式）")
    logger.info("=" * 60)

    # 创建模拟VL客户端
    class MockVLClient:
        """模拟的VL客户端（不需要真实API密钥）"""

        class ChatCompletions:
            def create(self, **kwargs):
                # 模拟VL响应：简单规则判断
                messages = kwargs.get('messages', [])
                if messages and messages[0].get('content'):
                    content_items = messages[0]['content']
                    text_content = next(
                        (item['text'] for item in content_items if item.get('type') == 'text'),
                        ''
                    )

                    # 简单规则：如果题目中提到"图"、"下图"等，则认为相关
                    is_related = any(keyword in text_content for keyword in ['图', '下图', '右图', '如图'])

                    result = {
                        "is_related": is_related,
                        "reason": "模拟判断：题目提到图形" if is_related else "模拟判断：题目未提到图形",
                        "confidence": 0.9 if is_related else 0.3
                    }

                    class MockResponse:
                        def __init__(self, result):
                            self.choices = [type('obj', (object,), {
                                'message': type('obj', (object,), {
                                    'content': json.dumps(result, ensure_ascii=False)
                                })
                            })]

                    return MockResponse(result)

        def __init__(self):
            self.chat = type('obj', (object,), {'completions': self.ChatCompletions()})

    # 创建配置
    config = ProcessingConfig()
    config.enable_cache = False  # 测试时禁用缓存

    # 创建合并器
    mock_client = MockVLClient()
    merger = QuestionImageMerger(mock_client, config)

    # 加载测试数据
    with open('output/result_data.json', 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    original_image = cv2.imread('data/shuxue/1.png')

    layout_blocks = result_data['layoutParsingResults'][0]['prunedResult']['parsing_res_list']

    # 使用之前拆分的结果
    text_blocks = test_question_splitter()
    image_blocks = [b for b in layout_blocks if b.get('block_label') == 'image']

    logger.info(f"\n找到 {len(text_blocks)} 个文本块")
    logger.info(f"找到 {len(image_blocks)} 个图像块")

    # 测试合并
    merged_questions = merger.merge_text_and_images(
        text_blocks[:5],  # 只测试前5个题目
        image_blocks,
        original_image
    )

    logger.info(f"\n合并结果: {len(merged_questions)} 个完整题目")

    for i, question in enumerate(merged_questions):
        logger.info(f"\n题目 {i+1}:")
        logger.info(f"  题号: {question.get('question_id')}")
        logger.info(f"  有配图: {question.get('has_image')}")
        logger.info(f"  图片数: {len(question.get('images', []))}")
        logger.info(f"  坐标: {question.get('bbox')}")

        if question.get('has_image'):
            for j, img in enumerate(question.get('images', [])):
                logger.info(f"    图片 {j+1}:")
                logger.info(f"      坐标: {img.get('block_bbox')}")
                logger.info(f"      VL验证: {img.get('vl_verification', {})}")

    return merged_questions


def test_multi_image_scenario():
    """测试一题多图场景"""
    logger.info("\n" + "=" * 60)
    logger.info("测试一题多图场景")
    logger.info("=" * 60)

    # 构造测试数据：一个题目下方有4张同行图片（模拟选择题ABCD）
    text_block = {
        'block_label': 'text',
        'block_content': '2. 被方框( )遮住一部分的图形可能是平行四边形。',
        'block_bbox': [3094, 2605, 4996, 2707],
        'question_number': 2
    }

    image_blocks = [
        {'block_label': 'image', 'block_bbox': [3386, 2824, 3594, 3061], 'block_id': 20},
        {'block_label': 'image', 'block_bbox': [3828, 2831, 4035, 3064], 'block_id': 21},
        {'block_label': 'image', 'block_bbox': [4270, 2834, 4482, 3067], 'block_id': 22},
        {'block_label': 'image', 'block_bbox': [4712, 2840, 4923, 3070], 'block_id': 23},
    ]

    # 创建模拟客户端
    class MockVLClient:
        class ChatCompletions:
            def create(self, **kwargs):
                # 模拟批量验证：选择题的4个选项图都相关
                result = {
                    "images": [
                        {"index": 0, "is_related": True, "reason": "选项A图形", "confidence": 0.95},
                        {"index": 1, "is_related": True, "reason": "选项B图形", "confidence": 0.95},
                        {"index": 2, "is_related": True, "reason": "选项C图形", "confidence": 0.95},
                        {"index": 3, "is_related": True, "reason": "选项D图形", "confidence": 0.95},
                    ]
                }

                class MockResponse:
                    def __init__(self, result):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': json.dumps(result, ensure_ascii=False)
                            })
                        })]

                return MockResponse(result)

        def __init__(self):
            self.chat = type('obj', (object,), {'completions': self.ChatCompletions()})

    config = ProcessingConfig()
    config.enable_cache = False

    merger = QuestionImageMerger(MockVLClient(), config)

    # 创建模拟图像
    original_image = np.zeros((5000, 7000, 3), dtype=np.uint8)

    # 测试合并
    merged = merger.merge_text_and_images([text_block], image_blocks, original_image)

    logger.info(f"\n合并结果:")
    logger.info(f"  题号: {merged[0].get('question_id')}")
    logger.info(f"  有配图: {merged[0].get('has_image')}")
    logger.info(f"  图片数: {len(merged[0].get('images', []))} ✅ 应为4")
    logger.info(f"  完整坐标: {merged[0].get('bbox')}")

    assert len(merged[0].get('images', [])) == 4, "一题多图测试失败！"
    logger.info("✅ 一题多图功能正常！")


def main():
    """运行所有测试"""
    logger.info("开始运行测试...")

    try:
        # 测试1: 题目拆分
        test_question_splitter()

        # 测试2: 图文合并（模拟模式）
        test_image_merger_mock()

        # 测试3: 一题多图
        test_multi_image_scenario()

        logger.info("\n" + "=" * 60)
        logger.info("✅ 所有测试通过！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
