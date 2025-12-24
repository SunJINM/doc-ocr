# 数学试卷题目提取系统

完整的题目分割优化解决方案，解决PP-OCR-VL检测结果中的两大问题：
1. **题目文本与配图分离** - 使用Qwen-VL视觉模型进行语义验证和合并
2. **多个题目被合并到一个块** - 通过题号识别和OCR定位进行智能拆分

## 核心特性

### ✅ 关键问题解决

- **先拆分，后合并**：正确的处理顺序确保准确性
- **一题多图支持**：完美处理选择题ABCD四个选项图等场景
- **批量VL验证**：一次API调用判断多个候选图像，提高效率
- **空间规则优先**：高置信度场景跳过VL调用，降低成本
- **智能降级策略**：VL失败时自动降级到空间规则

### 🎯 性能指标

| 指标 | 目标值 |
|-----|-------|
| 题目检测准确率 | ≥95% |
| 图文关联准确率 | ≥90% |
| 拆分准确率 | ≥92% |
| 处理速度 | ≤45秒/页 |
| 单份试卷成本 | ≈0.006元 |

## 快速开始

### 1. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements_extraction.txt

# 如果使用GPU（推荐）
pip install paddlepaddle-gpu

# 如果使用CPU
pip install paddlepaddle
```

**重要提示**：代码已适配PaddleOCR最新版本API，如遇到兼容性问题请查看 [CHANGELOG.md](CHANGELOG.md)

### 2. 配置API密钥

```bash
# 设置Qwen-VL API密钥
export DASHSCOPE_API_KEY="your-api-key-here"

# 或者在命令行指定
```

获取API密钥：https://dashscope.console.aliyun.com/

### 3. 运行提取

```bash
# 基础用法
python main_extraction.py \
  --result-data output/result_data.json \
  --image data/shuxue/1.png \
  --output output/structured_questions.json

# 指定API密钥
python main_extraction.py \
  --api-key "your-api-key" \
  --result-data output/result_data.json \
  --image data/shuxue/1.png
```

### 4. 查看结果

提取完成后会生成：
- `output/structured_questions.json` - 结构化题目数据
- `output/structured_questions_visualization.jpg` - 可视化结果图

## 输出格式

### 结构化JSON示例

```json
{
  "exam_info": {
    "title": "人教版 2025-2026 学年小学四年级数学上册期末",
    "subject": "数学",
    "grade": "四年级",
    "time_limit": "30分钟",
    "total_score": "50+5"
  },
  "questions": [
    {
      "id": 1,
      "type": "填空题",
      "score": 1,
      "content": {
        "text": "1. 学校组织 \"冰雪同梦，亚洲同心\" 系列活动...",
        "has_image": false,
        "image_count": 0,
        "images": []
      },
      "bbox": [131, 1165, 2996, 1437],
      "metadata": {
        "split_from_merged": false
      }
    },
    {
      "id": 2,
      "type": "填空题",
      "score": 1,
      "content": {
        "text": "2. 下图中有( )条线段、( )条直线、( )条射线。",
        "has_image": true,
        "image_count": 1,
        "images": [
          {
            "block_id": 5,
            "bbox": [1341, 1778, 1808, 2109],
            "vl_verified": true,
            "verification": {
              "is_related": true,
              "reason": "图片展示了几何图形",
              "confidence": 0.95
            }
          }
        ]
      },
      "bbox": [129, 1538, 1808, 2109],
      "metadata": {
        "split_from_merged": false,
        "image_count": 1
      }
    }
  ],
  "metadata": {
    "total_questions": 10,
    "with_images": 5,
    "total_images": 7,
    "split_count": 2,
    "processing_timestamp": "2025-12-24T10:00:00"
  }
}
```

## 模块说明

### 核心模块

```
src/question_extraction/
├── config.py              # 配置管理
├── question_splitter.py   # 题目拆分模块
├── question_merger.py     # 图文合并模块（支持一题多图）
├── extractor.py          # 主流程编排器
├── visualizer.py         # 可视化工具
└── evaluator.py          # 评估工具
```

### 处理流程

```
PP-OCR-VL检测结果
    ↓
步骤1: 题目拆分（QuestionSplitter）
    ├─ 题号检测（正则匹配）
    ├─ OCR精细定位
    └─ 边界框重计算
    ↓
步骤2: 图文合并（QuestionImageMerger）
    ├─ 空间位置筛选
    ├─ 支持一题多图（同行图像聚类）
    ├─ Qwen-VL验证（批量/单个）
    └─ 计算完整坐标框
    ↓
结构化题目数据
```

## 配置说明

### 关键配置项

编辑 `src/question_extraction/config.py`：

```python
@dataclass
class ProcessingConfig:
    # 图文合并配置
    max_vertical_distance: int = 300  # 图文最大垂直距离
    max_horizontal_distance: int = 200  # 图文最大水平距离

    # VL验证配置
    vl_confidence_threshold: float = 0.7  # VL验证置信度阈值
    enable_vl_batch: bool = True  # 启用批量VL验证
    vl_batch_size: int = 3

    # 一题多图配置
    max_images_per_question: int = 8  # 单题最多图片数
    image_clustering_enabled: bool = True  # 启用图像聚类
    same_row_threshold: int = 50  # 同行Y坐标差异阈值

    # 缓存配置
    enable_cache: bool = True  # 启用VL结果缓存
    cache_dir: str = 'cache/vl_results'
```

## 使用示例

### 示例1：基础提取

```python
from src.question_extraction import ExamPaperQuestionExtractor
from openai import OpenAI
from paddleocr import PaddleOCR

# 初始化
qwen_client = OpenAI(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 创建提取器
extractor = ExamPaperQuestionExtractor(qwen_client, ocr, config)

# 执行提取
result = extractor.extract_questions(
    'output/result_data.json',
    'data/shuxue/1.png'
)

# 查看结果
print(f"提取了 {result['metadata']['total_questions']} 个题目")
```

### 示例2：可视化

```python
from src.question_extraction import ResultVisualizer
import cv2

# 读取原始图像
original_image = cv2.imread('data/shuxue/1.png')

# 创建可视化
visualizer = ResultVisualizer()
visualizer.visualize_extraction_results(
    original_image,
    result['questions'],
    'output/visualization.jpg'
)
```

### 示例3：评估

```python
from src.question_extraction import ExtractionEvaluator

# 准备标注数据
ground_truth = [...]  # 人工标注的题目数据

# 评估
evaluator = ExtractionEvaluator()
metrics = evaluator.evaluate(result['questions'], ground_truth)

print(f"检测准确率: {metrics['detection_precision']:.2%}")
print(f"图文关联准确率: {metrics['image_association_acc']:.2%}")
```

## 处理特殊场景

### 场景1：一题多图（选择题）

系统自动识别同一行的多个图像：

```python
# 配置
processing_config.image_clustering_enabled = True
processing_config.same_row_threshold = 50  # Y坐标差异阈值

# 批量VL验证提高效率
processing_config.enable_vl_batch = True
```

### 场景2：题号不规范

支持多种题号格式：

```python
question_number_patterns = [
    r'^(\d+)[\.、]\s*',      # 1. 或 1、
    r'^\((\d+)\)\s*',        # (1)
    r'^第(\d+)题\s*',        # 第1题
    r'^\[(\d+)\]\s*',        # [1]
    r'^[【](\d+)[】]\s*',     # 【1】
]
```

### 场景3：降低成本

```python
# 策略1: 提高空间置信度阈值，减少VL调用
processing_config.spatial_confidence_threshold = 0.9

# 策略2: 启用缓存
processing_config.enable_cache = True

# 策略3: 批量验证
processing_config.enable_vl_batch = True
```

## 性能优化

### 成本优化

- **空间规则优先**：高置信度（>0.9）直接通过，不调用VL
- **批量验证**：一题多图时批量调用，减少API次数
- **结果缓存**：相同题目-图像组合复用缓存
- **预期成本**：单份试卷约0.006元

### 速度优化

- **并行处理**：独立题目并行拆分和验证
- **GPU加速**：OCR使用GPU提速
- **智能降级**：VL失败时快速降级到空间规则

## 常见问题

### Q1: Qwen-VL调用失败怎么办？

A: 系统会自动降级到空间规则，不影响基本功能。检查：
- API密钥是否正确
- 网络连接是否正常
- 查看日志了解具体错误

### Q2: 如何处理手写试卷？

A: 调整OCR配置：
```python
ocr_config.det_db_thresh = 0.2  # 降低检测阈值
# 使用手写识别模型
```

### Q3: 一题多图识别不准确？

A: 调整聚类参数：
```python
processing_config.same_row_threshold = 80  # 增大阈值
processing_config.max_images_per_question = 10  # 增加最大数量
```

### Q4: 题目拆分错误？

A: 检查题号格式，添加自定义模式：
```python
processing_config.question_number_patterns.append(
    r'^自定义模式'
)
```

## 技术文档

详细的技术方案请参考：
- [题目分割优化方案.md](docs/题目分割优化方案.md) - 完整技术方案
- [数学试卷版面分析与内容提取技术方案.md](docs/数学试卷版面分析与内容提取技术方案.md) - 整体架构

## 许可证

MIT License
