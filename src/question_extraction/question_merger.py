"""
图文合并模块

处理题目文本与配图的关联，支持一题多图场景
"""
import base64
import logging
import hashlib
import pickle
import os
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import json
from src.question_extraction.config import qwen_vl_config

logger = logging.getLogger(__name__)


class QuestionImageMerger:
    """题目与图像合并处理器"""

    def __init__(self, qwen_vl_client, config):
        """
        初始化图文合并器

        Args:
            qwen_vl_client: OpenAI SDK兼容的Qwen-VL客户端
            config: ProcessingConfig配置实例
        """
        self.vl_client = qwen_vl_client
        self.config = config

        # 创建缓存目录
        if config.enable_cache:
            os.makedirs(config.cache_dir, exist_ok=True)

    def merge_text_and_images(
        self,
        text_blocks: List[Dict[str, Any]],
        image_blocks: List[Dict[str, Any]],
        original_image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        合并文本题目和对应的图像（支持一题多图）

        Args:
            text_blocks: 文本块列表（已拆分的独立题目）
            image_blocks: 图像块列表
            original_image: 原始试卷图像

        Returns:
            merged_questions: 合并后的题目列表，包含完整坐标框
        """
        merged_questions = []
        used_images = set()  # 记录已使用的图像ID

        logger.info(f"开始合并：{len(text_blocks)}个文本块，{len(image_blocks)}个图像块")

        for text_block in text_blocks:
            # 步骤1: 空间关系预分析（找出所有候选图像）
            candidate_images = self._find_spatial_related_images(
                text_block,
                image_blocks,
                used_images
            )
            print(f"-----------{text_block}, {candidate_images}")

            if not candidate_images:
                # 无配图的题目
                merged_questions.append({
                    'question_id': text_block.get('question_number'),
                    'text_content': text_block.get('block_content', ''),
                    'has_image': False,
                    'images': [],
                    'bbox': text_block.get('block_bbox', []),
                    'components': [text_block],
                    'metadata': {
                        'split_from_merged': text_block.get('split_from_merged', False)
                    }
                })
                logger.debug(f"题目{text_block.get('question_number')}无配图")
                continue

            logger.info(f"题目{text_block.get('question_number')}找到{len(candidate_images)}个候选图像")

            # 步骤2: 使用Qwen-VL验证（支持批量）
            if self.config.enable_vl_batch and len(candidate_images) > 1:
                verified_images = self._batch_verify_with_qwen_vl(
                    text_block,
                    candidate_images,
                    original_image
                )
            else:
                verified_images = self._verify_with_qwen_vl(
                    text_block,
                    candidate_images,
                    original_image
                )

            # 步骤3: 标记已使用的图像
            for img in verified_images:
                used_images.add(img.get('block_id'))

            # 步骤4: 计算合并后的坐标框
            if verified_images:
                merged_bbox = self._calculate_merged_bbox(
                    text_block.get('block_bbox', []),
                    [img.get('block_bbox', []) for img in verified_images]
                )
            else:
                merged_bbox = text_block.get('block_bbox', [])

            # 步骤5: 构建完整题目结构
            merged_questions.append({
                'question_id': text_block.get('question_number'),
                'text_content': text_block.get('block_content', ''),
                'has_image': len(verified_images) > 0,
                'images': verified_images,
                'bbox': merged_bbox,
                'components': [text_block] + verified_images,
                'metadata': {
                    'split_from_merged': text_block.get('split_from_merged', False),
                    'image_count': len(verified_images),
                    'spatial_candidates': len(candidate_images)
                }
            })

            logger.info(f"题目{text_block.get('question_number')}验证通过{len(verified_images)}张图像")

        logger.info(f"合并完成：{len(merged_questions)}个题目")
        return merged_questions

    def _find_spatial_related_images(
        self,
        text_block: Dict[str, Any],
        image_blocks: List[Dict[str, Any]],
        used_images: set,
        max_vertical_distance: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        基于空间位置查找可能相关的图像（支持多图）

        策略：
        1. 图像在文本块下方或右侧
        2. 垂直/水平距离在阈值内
        3. 支持同一行的多个图像（如选择题ABCD四个选项图）
        """
        if max_vertical_distance is None:
            max_vertical_distance = self.config.max_vertical_distance

        text_bbox = text_block.get('block_bbox', [])
        if len(text_bbox) != 4:
            return []

        text_bottom = text_bbox[3]
        text_left = text_bbox[0]
        text_right = text_bbox[2]
        text_center_x = (text_left + text_right) / 2

        candidates = []

        for img_block in image_blocks:
            # 跳过已使用的图像
            if img_block.get('block_id') in used_images:
                continue

            img_bbox = img_block.get('block_bbox', [])
            if len(img_bbox) != 4:
                continue

            img_top = img_bbox[1]
            img_bottom = img_bbox[3]
            img_left = img_bbox[0]
            img_right = img_bbox[2]
            img_center_x = (img_left + img_right) / 2
            img_center_y = (img_top + img_bottom) / 2

            # 条件1: 图像在文本下方
            if img_top < text_bottom:
                # 检查是否在右侧（水平排列）
                horizontal_distance = img_left - text_right
                if 0 < horizontal_distance < self.config.max_horizontal_distance:
                    # 右侧图像
                    confidence = self._calculate_spatial_confidence(text_bbox, img_bbox, 'right')
                    candidates.append({
                        'block': img_block,
                        'distance': horizontal_distance,
                        'direction': 'right',
                        'confidence': confidence
                    })
                continue

            # 条件2: 垂直距离检查
            vertical_distance = img_top - text_bottom
            if vertical_distance > max_vertical_distance:
                continue

            # 条件3: 水平位置关联性检查
            horizontal_related = self._check_horizontal_relation(
                text_bbox, img_bbox
            )

            if horizontal_related:
                confidence = self._calculate_spatial_confidence(text_bbox, img_bbox, 'below')
                candidates.append({
                    'block': img_block,
                    'distance': vertical_distance,
                    'direction': 'below',
                    'confidence': confidence
                })

        # 按距离排序
        candidates.sort(key=lambda x: x['distance'])

        # 处理一题多图场景：检测同一行的多个图像
        grouped_candidates = self._group_images_by_row(candidates)

        return [c['block'] for c in grouped_candidates[:self.config.max_images_per_question]]

    def _check_horizontal_relation(
        self,
        text_bbox: List[int],
        img_bbox: List[int]
    ) -> bool:
        """检查文本和图像的水平关联性"""
        text_left, text_right = text_bbox[0], text_bbox[2]
        img_left, img_right = img_bbox[0], img_bbox[2]
        text_center_x = (text_left + text_right) / 2
        img_center_x = (img_left + img_right) / 2

        # 情况A: 图像中心在文本水平范围内
        if text_left <= img_center_x <= text_right:
            return True

        # 情况B: 文本中心在图像水平范围内
        if img_left <= text_center_x <= img_right:
            return True

        # 情况C: 两者有水平重叠
        if img_left <= text_right and img_right >= text_left:
            return True

        # 情况D: 距离不远（可能是右侧排列）
        if abs(img_center_x - text_center_x) < self.config.max_horizontal_distance:
            return True

        return False

    def _group_images_by_row(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        将同一行的图像聚合（如选择题ABCD四个选项图）

        策略：Y坐标接近的图像视为同一行
        """
        if not candidates or not self.config.image_clustering_enabled:
            return candidates

        grouped = []
        current_row = []
        last_y = None

        for candidate in candidates:
            img_bbox = candidate['block'].get('block_bbox', [])
            img_y = (img_bbox[1] + img_bbox[3]) / 2  # 中心Y坐标

            if last_y is None or abs(img_y - last_y) < self.config.same_row_threshold:
                # 同一行
                current_row.append(candidate)
                last_y = img_y
            else:
                # 新的一行
                grouped.extend(current_row)
                current_row = [candidate]
                last_y = img_y

        # 添加最后一行
        if current_row:
            grouped.extend(current_row)

        return grouped

    def _calculate_spatial_confidence(
        self,
        text_bbox: List[int],
        img_bbox: List[int],
        direction: str
    ) -> float:
        """
        计算空间关联的置信度

        考虑因素：
        1. 距离（越近越高）
        2. 水平对齐程度
        3. 大小比例
        """
        text_left, text_top, text_right, text_bottom = text_bbox
        img_left, img_top, img_right, img_bottom = img_bbox

        if direction == 'below':
            # 垂直距离
            distance = img_top - text_bottom
            distance_score = max(0, 1 - distance / self.config.max_vertical_distance)

            # 水平对齐度
            text_center_x = (text_left + text_right) / 2
            img_center_x = (img_left + img_right) / 2
            alignment_score = max(0, 1 - abs(text_center_x - img_center_x) / (text_right - text_left))

        else:  # right
            # 水平距离
            distance = img_left - text_right
            distance_score = max(0, 1 - distance / self.config.max_horizontal_distance)

            # 垂直对齐度
            text_center_y = (text_top + text_bottom) / 2
            img_center_y = (img_top + img_bottom) / 2
            alignment_score = max(0, 1 - abs(text_center_y - img_center_y) / (text_bottom - text_top))

        # 综合得分
        confidence = 0.6 * distance_score + 0.4 * alignment_score

        return confidence

    def _verify_with_qwen_vl(
        self,
        text_block: Dict[str, Any],
        candidate_images: List[Dict[str, Any]],
        original_image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        使用Qwen-VL验证图像与题目的关联性（逐个验证）

        通过视觉语言模型理解题目内容和图像内容的语义关系
        """
        verified_images = []
        question_text = text_block.get('block_content', '')

        for img_block in candidate_images:
            # 检查缓存
            if self.config.enable_cache:
                cached_result = self._get_cached_result(question_text, img_block, original_image)
                if cached_result is not None:
                    if cached_result['is_related']:
                        img_block['vl_verification'] = cached_result
                        verified_images.append(img_block)
                    continue

            # 裁剪图像区域
            img_bbox = img_block.get('block_bbox', [])
            cropped_image = self._crop_image(original_image, img_bbox)

            # 调用VL模型
            try:
                result = self._call_vl_model(question_text, cropped_image)

                # 缓存结果
                if self.config.enable_cache:
                    self._cache_result(question_text, img_block, original_image, result)

                if result['is_related'] and result['confidence'] > self.config.vl_confidence_threshold:
                    img_block['vl_verification'] = result
                    verified_images.append(img_block)

            except Exception as e:
                logger.error(f"VL验证失败: {e}")
                continue

        return verified_images

    def _batch_verify_with_qwen_vl(
        self,
        text_block: Dict[str, Any],
        candidate_images: List[Dict[str, Any]],
        original_image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        批量验证多个候选图像（一次API调用判断所有图像）

        适用于一题多图场景，更高效
        """
        question_text = text_block.get('block_content', '')

        # 裁剪所有候选图像
        cropped_images = []
        for img_block in candidate_images:
            img_bbox = img_block.get('block_bbox', [])
            cropped_image = self._crop_image(original_image, img_bbox)
            cropped_images.append(cropped_image)

        try:
            # 构建批量提示词
            prompt = f"""
请分析以下数学题目需要哪些图片来辅助解答：

题目内容：
{question_text}

我将提供{len(cropped_images)}张候选图片，请判断每张图片是否与题目相关。

请以JSON格式返回结果，格式如下：
{{
    "images": [
        {{"index": 0, "is_related": true/false, "reason": "说明", "confidence": 0.0-1.0}},
        {{"index": 1, "is_related": true/false, "reason": "说明", "confidence": 0.0-1.0}},
        ...
    ]
}}
"""

            # 构建消息内容（文本 + 多张图片）
            content = [{"type": "text", "text": prompt}]

            for i, cropped_img in enumerate(cropped_images):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_to_base64_url(cropped_img)
                    }
                })

            # 调用VL模型
            response = self.vl_client.chat.completions.create(
                model=qwen_vl_config.model,
                messages=[{"role": "user", "content": content}],
                temperature=qwen_vl_config.temperature,
                max_tokens=qwen_vl_config.max_tokens
            )

            # 解析响应
            result_text = response.choices[0].message.content.strip()

            # 尝试从响应中提取JSON（可能包含额外文本）
            try:
                # 尝试直接解析
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # 尝试提取JSON代码块
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # 尝试查找花括号包裹的JSON
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                    else:
                        raise ValueError(f"无法从响应中提取JSON: {result_text[:200]}")

            # 构建验证结果
            verified_images = []

            for img_result in result.get('images', []):
                idx = img_result.get('index', -1)
                if 0 <= idx < len(candidate_images):
                    if img_result.get('is_related') and \
                       img_result.get('confidence', 0) > self.config.vl_confidence_threshold:
                        img_block = candidate_images[idx]
                        img_block['vl_verification'] = img_result
                        verified_images.append(img_block)

            return verified_images

        except Exception as e:
            logger.error(f"批量VL验证失败，降级为逐个验证: {e}")
            # 降级策略：逐个验证
            return self._verify_with_qwen_vl(text_block, candidate_images, original_image)

    def _call_vl_model(self, question_text: str, image: np.ndarray) -> Dict[str, Any]:
        """调用Qwen-VL模型（单图）"""
        prompt = f"""
请分析以下数学题目是否需要这张图片来辅助解答：

题目内容：
{question_text}

请回答：
1. 这张图片是否与题目直接相关？（是/否）
2. 如果相关，图片在题目中的作用是什么？

请以JSON格式回答：
{{
    "is_related": true/false,
    "reason": "原因说明",
    "confidence": 0.0-1.0
}}
"""

        response = self.vl_client.chat.completions.create(
            model=qwen_vl_config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_base64_url(image)
                            }
                        }
                    ]
                }
            ],
            temperature=qwen_vl_config.temperature,
            max_tokens=qwen_vl_config.max_tokens
        )

        result_text = response.choices[0].message.content.strip()

        # 尝试从响应中提取JSON（可能包含额外文本）
        try:
            # 尝试直接解析
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # 尝试提取JSON代码块
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # 尝试查找花括号包裹的JSON
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    logger.error(f"无法从VL响应中提取JSON: {result_text[:200]}")
                    # 返回默认值
                    result = {
                        "is_related": False,
                        "reason": "解析失败",
                        "confidence": 0.0
                    }

        return result

    def _calculate_merged_bbox(
        self,
        text_bbox: List[int],
        image_bboxes: List[List[int]]
    ) -> List[int]:
        """计算合并后的完整题目坐标框（最小外接矩形）"""
        all_bboxes = [text_bbox] + image_bboxes

        x1 = min(bbox[0] for bbox in all_bboxes if len(bbox) == 4)
        y1 = min(bbox[1] for bbox in all_bboxes if len(bbox) == 4)
        x2 = max(bbox[2] for bbox in all_bboxes if len(bbox) == 4)
        y2 = max(bbox[3] for bbox in all_bboxes if len(bbox) == 4)

        return [x1, y1, x2, y2]

    def _crop_image(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """裁剪图像区域"""
        x1, y1, x2, y2 = [int(c) for c in bbox]
        return image[y1:y2, x1:x2]

    def _image_to_base64_url(self, image: np.ndarray) -> str:
        """将图像转换为base64 URL"""
        _, buffer = cv2.imencode('.jpg', image)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"

    def _get_cache_key(self, text: str, img_block: Dict[str, Any], image: np.ndarray) -> str:
        """生成缓存键"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        bbox_str = str(img_block.get('block_bbox', []))
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()
        return f"{text_hash}_{bbox_hash}"

    def _get_cached_result(
        self,
        text: str,
        img_block: Dict[str, Any],
        image: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        cache_key = self._get_cache_key(text, img_block, image)
        cache_path = os.path.join(self.config.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")

        return None

    def _cache_result(
        self,
        text: str,
        img_block: Dict[str, Any],
        image: np.ndarray,
        result: Dict[str, Any]
    ):
        """缓存结果"""
        cache_key = self._get_cache_key(text, img_block, image)
        cache_path = os.path.join(self.config.cache_dir, f"{cache_key}.pkl")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
