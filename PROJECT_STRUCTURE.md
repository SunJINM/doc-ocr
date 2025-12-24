# 项目结构说明

## 📁 完整目录结构

```
doc-ocr/
│
├── 📄 main_extraction.py              # 主程序入口
├── 📄 test_extraction.py              # 测试脚本（无需API密钥）
│
├── 📚 文档
│   ├── README.md                      # 项目总览
│   ├── README_EXTRACTION.md           # 题目提取模块完整文档
│   ├── QUICKSTART.md                  # 快速入门指南
│   ├── PROJECT_STRUCTURE.md           # 本文件
│   └── IFLOW.md                       # 工作流说明
│
├── 📂 docs/                           # 技术文档目录
│   ├── 题目分割优化方案.md             # 详细技术方案
│   ├── 数学试卷版面分析与内容提取技术方案.md
│   ├── PaddleOCR版面检测技术总结.md
│   ├── 版面分析模块技术总结.md
│   └── 公式识别模块技术总结.md
│
├── 📂 src/                            # 源代码目录
│   └── question_extraction/           # 题目提取模块
│       ├── __init__.py                # 模块初始化
│       ├── config.py                  # 配置管理
│       ├── question_splitter.py       # 题目拆分（先拆分）
│       ├── question_merger.py         # 图文合并（支持一题多图）
│       ├── extractor.py               # 主流程编排
│       ├── visualizer.py              # 可视化工具
│       └── evaluator.py               # 评估工具
│
├── 📂 data/                           # 数据目录
│   └── shuxue/                        # 数学试卷示例
│       └── 1.png                      # 示例试卷图像
│
├── 📂 output/                         # 输出目录
│   ├── result_data.json               # PP-OCR-VL原始结果
│   ├── structured_questions.json      # 提取的结构化题目
│   ├── structured_questions_visualization.jpg  # 可视化结果
│   ├── doc_0.md                       # Markdown格式输出
│   ├── imgs/                          # 提取的图像
│   └── *.jpg                          # 版面检测可视化
│
├── 📂 cache/                          # 缓存目录
│   └── vl_results/                    # VL验证结果缓存
│
├── 📂 tests/                          # 原始测试脚本
│   ├── pp_ocr_vl.py                   # PP-OCR-VL调用示例
│   ├── layout_pp_ocrv5.py
│   └── pp_structurev3_test.py
│
├── 📄 requirements_extraction.txt     # 依赖列表
├── 📄 check.py                        # 检查脚本
└── 📄 extraction.log                  # 运行日志

```

## 🔑 核心模块说明

### 1. 主程序 (`main_extraction.py`)

**功能**：完整的题目提取流程编排

**核心流程**：
1. 加载PP-OCR-VL检测结果和原始图像
2. 先拆分合并的题目块（QuestionSplitter）
3. 再合并题目文本与配图（QuestionImageMerger）
4. 生成结构化数据和可视化结果

**使用**：
```bash
python main_extraction.py \
  --result-data output/result_data.json \
  --image data/shuxue/1.png
```

### 2. 题目拆分模块 (`question_splitter.py`)

**功能**：处理一个文本块包含多个题目的情况

**核心算法**：
- 题号检测（正则匹配多种格式）
- OCR精细定位题号坐标
- 边界框智能重计算

**关键方法**：
```python
def split_merged_questions(text_block, original_image):
    """
    输入：包含多个题目的文本块
    输出：拆分后的独立题目列表
    """
```

**支持的题号格式**：
- `1.` / `1、` - 阿拉伯数字+标点
- `(1)` - 括号格式
- `第1题` - 中文格式
- `[1]` - 方括号格式
- `【1】` - 全角方括号

### 3. 图文合并模块 (`question_merger.py`)

**功能**：合并题目文本与配图，支持一题多图

**核心算法**：
- 空间位置预筛选（快速）
- Qwen-VL语义验证（准确）
- 同行图像聚类（一题多图）
- 批量VL验证（高效）

**关键方法**：
```python
def merge_text_and_images(text_blocks, image_blocks, original_image):
    """
    输入：文本块列表 + 图像块列表
    输出：完整题目列表（包含关联的图像）
    """
```

**一题多图支持**：
```python
# 自动识别同一行的多个图像（如选择题ABCD）
def _group_images_by_row(candidates):
    """按Y坐标聚类同行图像"""
```

### 4. 主流程编排 (`extractor.py`)

**功能**：整合拆分和合并流程

**处理顺序**（重要）：
```python
1. 加载数据
2. 先拆分题目 ← 第一步
3. 再合并图文 ← 第二步
4. 构建结构化数据
```

**输出格式**：
```json
{
  "exam_info": {...},
  "questions": [...],
  "metadata": {
    "total_questions": 10,
    "with_images": 5,
    "total_images": 7,
    "split_count": 2
  }
}
```

### 5. 可视化工具 (`visualizer.py`)

**功能**：生成带标注的可视化图片

**可视化内容**：
- 题目边界框（不同颜色表示不同状态）
- 题号和类型标注
- 图像框和关联线
- 拆分标记
- VL验证标记
- 统计信息面板

**颜色方案**：
- 🟢 绿色：带配图的题目
- 🔵 蓝色：无配图的题目
- 🟡 黄色：图像块
- 🔷 青色：图文关联线
- 🟠 橙色：拆分的题目
- 🔴 红点：VL验证通过

### 6. 配置管理 (`config.py`)

**三大配置类**：

1. **QwenVLConfig** - Qwen-VL模型配置
   ```python
   api_key: str          # API密钥
   model: str            # 模型名称（qwen-vl-plus/max）
   temperature: float    # 温度参数
   ```

2. **OCRConfig** - PaddleOCR配置
   ```python
   use_gpu: bool         # 是否使用GPU
   det_db_thresh: float  # 检测阈值
   ```

3. **ProcessingConfig** - 处理配置
   ```python
   # 图文合并
   max_vertical_distance: int = 300
   vl_confidence_threshold: float = 0.7

   # 一题多图
   max_images_per_question: int = 8
   image_clustering_enabled: bool = True
   same_row_threshold: int = 50

   # 性能优化
   enable_cache: bool = True
   enable_vl_batch: bool = True
   ```

### 7. 评估工具 (`evaluator.py`)

**评估指标**：
- 题目检测准确率（精确率/召回率/F1）
- 边界框IoU（交并比）
- 图文关联准确率
- 拆分准确率

## 📊 数据流

```
原始试卷图像
    ↓
PP-OCR-VL检测
    ↓
result_data.json (原始检测结果)
    ↓
题目拆分 (QuestionSplitter)
    ├─ 检测题号
    ├─ OCR定位
    └─ 边界重计算
    ↓
拆分后的文本块列表
    ↓
图文合并 (QuestionImageMerger)
    ├─ 空间筛选
    ├─ 一题多图聚类
    ├─ VL验证（批量/单个）
    └─ 计算完整坐标
    ↓
structured_questions.json (结构化题目)
    ↓
可视化生成 (ResultVisualizer)
    ↓
visualization.jpg (可视化结果)
```

## 🔧 配置文件位置

- **全局配置**：`src/question_extraction/config.py`
- **日志配置**：`main_extraction.py` 中的 logging.basicConfig
- **API密钥**：环境变量 `DASHSCOPE_API_KEY`

## 📝 日志文件

- **运行日志**：`extraction.log`
- **日志级别**：INFO
- **日志内容**：
  - 流程进度
  - 题目拆分详情
  - 图文合并详情
  - VL验证结果
  - 错误和警告

## 🧪 测试文件

### test_extraction.py

**功能**：无需API密钥的模拟测试

**测试内容**：
1. `test_question_splitter()` - 题目拆分功能
2. `test_image_merger_mock()` - 图文合并（模拟VL）
3. `test_multi_image_scenario()` - 一题多图场景

**运行**：
```bash
python test_extraction.py
```

## 📦 依赖管理

### requirements_extraction.txt

**核心依赖**：
- `opencv-python` - 图像处理
- `numpy` - 数组运算
- `paddlepaddle-gpu` - 深度学习框架
- `paddleocr` - OCR引擎
- `openai` - Qwen-VL客户端

**安装**：
```bash
pip install -r requirements_extraction.txt
```

## 🚀 快速定位

| 需求 | 文件 |
|------|------|
| 开始使用 | `QUICKSTART.md` |
| 完整文档 | `README_EXTRACTION.md` |
| 技术方案 | `docs/题目分割优化方案.md` |
| 主程序 | `main_extraction.py` |
| 测试 | `test_extraction.py` |
| 配置调整 | `src/question_extraction/config.py` |
| 题目拆分逻辑 | `src/question_extraction/question_splitter.py` |
| 图文合并逻辑 | `src/question_extraction/question_merger.py` |
| 一题多图处理 | `question_merger.py` 中的 `_group_images_by_row` |
| 可视化定制 | `src/question_extraction/visualizer.py` |

## 💡 开发建议

1. **修改配置**：从 `config.py` 开始
2. **添加题号格式**：修改 `question_number_patterns`
3. **调整VL阈值**：修改 `vl_confidence_threshold`
4. **优化一题多图**：调整 `same_row_threshold`
5. **自定义可视化**：修改 `visualizer.py` 中的颜色和标注

## 🎯 扩展方向

- **多学科支持**：在 `extractor.py` 中添加学科特定逻辑
- **复杂题型**：扩展 `_identify_question_type` 方法
- **多语言**：调整OCR配置支持其他语言
- **Web界面**：基于 `extractor.py` 构建API服务

## 📞 支持

- **技术问题**：查看 `extraction.log` 日志
- **配置问题**：参考 `README_EXTRACTION.md` 常见问题
- **使用问题**：参考 `QUICKSTART.md` 快速入门
