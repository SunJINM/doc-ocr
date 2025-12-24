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


def main():
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数学试卷题目提取')
    parser.add_argument(
        '--result-data',
        type=str,
        default='output/result_data1.json',
        help='PP-OCR-VL检测结果JSON路径'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='data/shuxue/mifeng_1.jpg',
        help='原始试卷图像路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/structured_questions1.json',
        help='输出结构化题目JSON路径'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='生成可视化结果'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Qwen-VL API密钥（或通过环境变量DASHSCOPE_API_KEY设置）'
    )

    args = parser.parse_args()

    # 1. 配置初始化
    logger.info("=" * 80)
    logger.info("数学试卷题目提取系统启动")
    logger.info("=" * 80)

    # Qwen-VL配置
    qwen_config = QwenVLConfig()
    if args.api_key:
        qwen_config.api_key = args.api_key

    if not qwen_config.api_key:
        logger.error("未设置Qwen-VL API密钥！")
        logger.error("请通过--api-key参数或环境变量DASHSCOPE_API_KEY设置")
        return

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

    # 2. 检查输入文件
    if not os.path.exists(args.result_data):
        logger.error(f"检测结果文件不存在: {args.result_data}")
        return

    if not os.path.exists(args.image):
        logger.error(f"图像文件不存在: {args.image}")
        return

    logger.info(f"输入文件:")
    logger.info(f"  检测结果: {args.result_data}")
    logger.info(f"  原始图像: {args.image}")

    # 3. 创建提取器
    extractor = ExamPaperQuestionExtractor(
        qwen_client,
        ocr,
        processing_config
    )

    # 4. 执行提取
    logger.info("\n" + "=" * 80)
    try:
        structured_exam = extractor.extract_questions(
            args.result_data,
            args.image
        )
    except Exception as e:
        logger.error(f"提取过程发生错误: {e}", exc_info=True)
        return

    # 5. 保存结果
    logger.info("\n" + "=" * 80)
    logger.info("保存提取结果")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(structured_exam, f, ensure_ascii=False, indent=2)

    logger.info(f"结构化题目数据已保存: {args.output}")

    # 6. 生成可视化
    if args.visualize:
        logger.info("\n" + "=" * 80)
        logger.info("生成可视化结果")

        original_image = cv2.imread(args.image)
        visualizer = ResultVisualizer()

        vis_output = args.output.replace('.json', '_visualization.jpg')

        try:
            visualizer.visualize_extraction_results(
                original_image,
                structured_exam['questions'],
                vis_output
            )
        except Exception as e:
            logger.error(f"可视化生成失败: {e}", exc_info=True)

    # 7. 输出统计摘要
    logger.info("\n" + "=" * 80)
    logger.info("提取结果摘要")
    logger.info("=" * 80)

    print(f"\n试卷信息:")
    print(f"  标题: {structured_exam['exam_info'].get('title', 'N/A')}")
    print(f"  学科: {structured_exam['exam_info'].get('subject', 'N/A')}")
    print(f"  年级: {structured_exam['exam_info'].get('grade', 'N/A')}")
    print(f"  时间: {structured_exam['exam_info'].get('time_limit', 'N/A')}")
    print(f"  满分: {structured_exam['exam_info'].get('total_score', 'N/A')}")

    print(f"\n题目统计:")
    print(f"  总题目数: {structured_exam['metadata']['total_questions']}")
    print(f"  带配图题目: {structured_exam['metadata']['with_images']}")
    print(f"  总图片数: {structured_exam['metadata']['total_images']}")
    print(f"  拆分题目数: {structured_exam['metadata']['split_count']}")

    # 题目类型分布
    question_types = {}
    for q in structured_exam['questions']:
        qtype = q['type']
        question_types[qtype] = question_types.get(qtype, 0) + 1

    print(f"\n题目类型分布:")
    for qtype, count in sorted(question_types.items()):
        print(f"  {qtype}: {count}")

    # 一题多图统计
    multi_image_questions = [
        q for q in structured_exam['questions']
        if q.get('content', {}).get('image_count', 0) > 1
    ]

    if multi_image_questions:
        print(f"\n一题多图题目: {len(multi_image_questions)}")
        for q in multi_image_questions:
            print(f"  题目{q['id']}: {q['content']['image_count']}张图片")

    print(f"\n输出文件:")
    print(f"  结构化数据: {args.output}")
    if args.visualize:
        print(f"  可视化图片: {vis_output}")

    logger.info("\n" + "=" * 80)
    logger.info("提取流程完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
