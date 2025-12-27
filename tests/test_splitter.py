import re
import logging

# 初始化日志（模拟代码环境）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 复刻核心检测逻辑
QUESTION_PATTERNS = [
    r'(\d+)[\.、]\s*',        # 1. 或 1、
    r'\((\d+)\)\s*',          # (1)
    r'第(\d+)题\s*',          # 第1题
    r'\[(\d+)\]\s*',          # [1]
    r'[【](\d+)[】]\s*',       # 【1】
]

def _detect_question_numbers(text: str, strict_line_start: bool = True):
    question_numbers = []
    for pattern in QUESTION_PATTERNS:
        try:
            flags = re.MULTILINE if strict_line_start else 0
            for match in re.finditer(pattern, text, flags):
                number = int(match.group(1))
                position = match.start()
                matched_str = match.group(0)
                question_numbers.append({
                    'number': number,
                    'position': position,
                    'matched_str': matched_str,
                    'pattern': pattern
                })
        except Exception as e:
            logger.warning(f"模式匹配失败 {pattern}: {e}")
            continue
    # 排序+去重（简化版）
    question_numbers.sort(key=lambda x: x['position'])
    filtered = []
    last_pos = -10
    for qn in question_numbers:
        if qn['position'] - last_pos > 5:
            filtered.append(qn)
            last_pos = qn['position']
    return filtered

# 测试文本（去掉外层引号，模拟实际输入）
test_text = "3.找一个点D，使四边形ABCD是一个等腰梯形，画出这个等腰梯形。(2分)4.在这个等腰梯形中画一条线段，将其分成一个平行四边形与一个三角形。(2分)五、解决问题。(共13分)"

# 无首行限制检测
result_loose = _detect_question_numbers(test_text, strict_line_start=True)
print("无首行限制检测结果：")
for item in result_loose:
    print(f"题号：{item['number']}，匹配字符串：{item['matched_str']}，位置：{item['position']}")

print(f"\n检测到的题号数量：{len(result_loose)}")