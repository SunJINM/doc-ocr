import os
import json
import re
import cv2
import base64
from openai import OpenAI
from typing import List, Dict, Any


class ExamPaperParser:
    """试卷题目拆分解析器"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://ark.cn-beijing.volces.com/api/v3"):
        """
        初始化解析器
        
        Args:
            api_key: API密钥，如果为None则从环境变量ARK_API_KEY读取
            base_url: API基础URL
        """
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        )
        # self.model = "ep-20251025164648-d66ns"
        self.model = "qwen-vl-plus"
        
    def encode_image(self, image_path: str) -> str:
        """
        将图片编码为base64字符串
        
        Args:
            image_path: 图片路径
            
        Returns:
            base64编码的图片字符串
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def create_prompt(self) -> str:
        """
        创建提示词
        
        Returns:
            提示词字符串
        """
        return """你是一个专业的试卷分析助手。请仔细分析这张试卷图片，识别并定位每道题目。

任务要求：
1. 识别所有题目，包括题目本身、配图、选项、作答区域
2. 对于每道题目，输出题目的完整边界框
3. 识别每道题目中的所有填空位置（横线、括号、方框、空白区域等）
4. 对于跨页的题目，输出多个边界框区域

输出格式（严格按照JSON格式）：
{
  "paper_info": {
    "total_questions": 题目总数
  },
  "questions": [
    {
      "question_id": 题号（整数）,
      "question_type": "题型（填空题/选择题/计算题/应用题/判断题/解答题）",
      "question_text": "题目完整文本内容",
      "question_bboxes": [
        "<bbox>x_min y_min x_max y_max</bbox>"
      ],
      "blanks": [
        {
          "blank_id": 填空序号（整数）,
          "blank_bbox": "<bbox>x_min y_min x_max y_max</bbox>",
          "blank_type": "填空类型（横线/小括号/中括号/方框/空白区域）"
        }
      ]
    }
  ]
}

注意事项：
1. 坐标格式必须是 <bbox>x_min y_min x_max y_max</bbox>，坐标值范围0-1000
2. 题目边界框必须完整包含题干、图片、选项、作答区域
3. 填空位置包括但不限于：下划线（___）、括号（）、方框□、空白横线区域
4. 对于跨页题目，question_bboxes数组包含多个bbox
5. 如果题目没有填空，blanks为空数组
6. 按照题号顺序输出
7. 确保每道题目的边界框不遗漏任何部分

请严格按照上述JSON格式输出，不要添加任何其他内容。"""
    
    def parse_bbox(self, bbox_str: str) -> List[int]:
        """
        解析bbox字符串，提取坐标
        
        Args:
            bbox_str: bbox字符串，格式如 "<bbox>175 98 791 476</bbox>"
            
        Returns:
            坐标列表 [x_min, y_min, x_max, y_max]
        """
        pattern = r'<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>'
        match = re.search(pattern, bbox_str)
        if match:
            return [int(match.group(i)) for i in range(1, 5)]
        return None
    
    def normalize_coordinates(self, coords: List[int], image_width: int, image_height: int) -> List[int]:
        """
        将归一化坐标转换为实际坐标
        
        Args:
            coords: 归一化坐标 [x_min, y_min, x_max, y_max]，范围0-1000
            image_width: 图片实际宽度
            image_height: 图片实际高度
            
        Returns:
            实际坐标 [x_min, y_min, x_max, y_max]
        """
        x_min, y_min, x_max, y_max = coords
        return [
            int(x_min * image_width / 1000),
            int(y_min * image_height / 1000),
            int(x_max * image_width / 1000),
            int(y_max * image_height / 1000)
        ]
    
    def clean_json_response(self, response_text: str) -> str:
        """
        清理响应文本，提取JSON内容
        
        Args:
            response_text: 原始响应文本
            
        Returns:
            清理后的JSON字符串
        """
        # 移除markdown代码块标记
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        return response_text
    
    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析模型返回的JSON响应
        
        Args:
            response_text: 响应文本
            
        Returns:
            解析后的字典
        """
        # 清理响应文本
        cleaned_text = self.clean_json_response(response_text)
        
        # 解析JSON
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始文本: {cleaned_text[:500]}...")
            raise
        
        # 处理bbox字符串，转换为坐标数组
        for question in data.get('questions', []):
            # 处理题目边界框
            parsed_bboxes = []
            for bbox_str in question.get('question_bboxes', []):
                coords = self.parse_bbox(bbox_str)
                if coords:
                    parsed_bboxes.append(coords)
            question['question_bboxes'] = parsed_bboxes
            
            # 处理填空边界框
            for blank in question.get('blanks', []):
                bbox_str = blank.get('blank_bbox', '')
                coords = self.parse_bbox(bbox_str)
                if coords:
                    blank['blank_bbox'] = coords
        
        return data
    
    def parse_image(self, image_path: str) -> Dict[str, Any]:
        """
        解析试卷图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            解析结果字典
        """
        # 编码图片
        base64_image = self.encode_image(image_path)
        
        # 创建提示词
        prompt = self.create_prompt()
        
        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        # 解析响应
        response_text = response.choices[0].message.content
        result = self.parse_json_response(response_text)
        
        return result
    
    def draw_annotations(self, image_path: str, parse_result: Dict[str, Any], output_path: str):
        """
        在图片上绘制标注
        
        Args:
            image_path: 原始图片路径
            parse_result: 解析结果
            output_path: 输出图片路径
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        height, width = image.shape[:2]
        
        # 遍历每道题目
        for question in parse_result.get('questions', []):
            question_id = question.get('question_id')
            
            # 绘制题目边界框（红色）
            for i, bbox in enumerate(question.get('question_bboxes', [])):
                coords = self.normalize_coordinates(bbox, width, height)
                x_min, y_min, x_max, y_max = coords
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                
                # 标注题号
                label = f"Q{question_id}" + (f"-{i+1}" if len(question.get('question_bboxes', [])) > 1 else "")
                cv2.putText(image, label, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 绘制填空位置（蓝色）
            for blank in question.get('blanks', []):
                blank_id = blank.get('blank_id')
                bbox = blank.get('blank_bbox')
                if bbox:
                    coords = self.normalize_coordinates(bbox, width, height)
                    x_min, y_min, x_max, y_max = coords
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    
                    # 标注填空编号
                    label = f"{blank_id}"
                    cv2.putText(image, label, (x_min, y_min - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 保存图片
        cv2.imwrite(output_path, image)
        print(f"标注图片已保存: {output_path}")
    
    def process(self, image_path: str, output_json_path: str = None, output_image_path: str = None):
        """
        处理试卷图片，输出JSON和标注图片
        
        Args:
            image_path: 输入图片路径
            output_json_path: 输出JSON路径，默认为原图片路径_result.json
            output_image_path: 输出图片路径，默认为原图片路径_annotated.png
        """
        # 设置默认输出路径
        if output_json_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_json_path = f"{base_name}_result.json"
        
        if output_image_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_image_path = f"{base_name}_annotated.png"
        
        # 解析图片
        print("正在解析试卷图片...")
        result = self.parse_image(image_path)
        
        # 保存JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"解析结果已保存: {output_json_path}")
        
        # 绘制标注
        print("正在绘制标注...")
        self.draw_annotations(image_path, result, output_image_path)
        
        return result


# 使用示例
if __name__ == "__main__":
    # 创建解析器
    # parser = ExamPaperParser(api_key="14ebfc74-500c-46d5-a58b-61ac61341018")
    parser = ExamPaperParser(api_key="sk-f436e171e65c4999bb7e8203f0862317", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 处理试卷图片
    image_path = "./data/shuxue/1.png"  # 替换为实际图片路径
    result = parser.process(
        image_path=image_path,
        output_json_path="./output/exam_result.json",
        output_image_path="./output/exam_annotated.png"
    )
    
    # 打印结果统计
    print(f"\n解析完成！")
    print(f"总题目数: {result['paper_info']['total_questions']}")
    for q in result['questions']:
        print(f"题目{q['question_id']}: {q['question_type']}, "
              f"区域数: {len(q['question_bboxes'])}, "
              f"填空数: {len(q['blanks'])}")