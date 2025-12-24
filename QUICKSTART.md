# 快速入门指南

## 🚀 5分钟快速开始

### 步骤1: 安装依赖

```bash
pip install -r requirements_extraction.txt
```

### 步骤2: 设置API密钥

```bash
# Linux/Mac
export DASHSCOPE_API_KEY="your-api-key-here"

# Windows
set DASHSCOPE_API_KEY=your-api-key-here
```

获取API密钥：https://dashscope.console.aliyun.com/

### 步骤3: 运行测试（可选）

```bash
# 无需API密钥的模拟测试
python test_extraction.py
```

### 步骤4: 执行完整提取

```bash
python main_extraction.py \
  --result-data output/result_data.json \
  --image data/shuxue/1.png \
  --output output/structured_questions.json
```

### 步骤5: 查看结果

- **结构化数据**: `output/structured_questions.json`
- **可视化图片**: `output/structured_questions_visualization.jpg`

## 📊 输出示例

提取完成后会显示：

```
试卷信息:
  标题: 人教版 2025-2026 学年小学四年级数学上册期末
  学科: 数学
  年级: 四年级
  时间: 30分钟
  满分: 50+5

题目统计:
  总题目数: 10
  带配图题目: 5
  总图片数: 7
  拆分题目数: 2

题目类型分布:
  填空题: 6
  判断题: 4

一题多图题目: 1
  题目2: 4张图片
```

## 🔧 关键功能演示

### 功能1: 拆分合并的题目

**问题**：第5题和第6题被合并到一个文本块

**解决**：
```
原始: 1个文本块包含2个题目
  ↓
拆分后: 2个独立题目
  - 题目5: [124, 3485, 2510, 3750]
  - 题目6: [124, 3750, 2510, 4008]
```

### 功能2: 合并题目与配图

**问题**：题目文本和配图是分开的block

**解决**：
```
题目2: "下图中有( )条线段..."
  + 图像block_id=5
  ↓
完整题目: bbox=[129, 1538, 1808, 2109]
  - 文本: [129, 1538, 2197, 1643]
  - 图像: [1341, 1778, 1808, 2109]
  - VL验证: ✅ confidence=0.95
```

### 功能3: 一题多图支持

**场景**：选择题ABCD四个选项都是图片

**处理**：
```
题目: "被方框遮住的图形可能是..."
  ↓
批量VL验证4张候选图像
  ↓
结果: 4张图像全部关联
  - 选项A: block_id=20 ✅
  - 选项B: block_id=21 ✅
  - 选项C: block_id=22 ✅
  - 选项D: block_id=23 ✅
```

## ⚙️ 配置调整

### 降低成本

编辑 `src/question_extraction/config.py`：

```python
# 提高空间置信度阈值，减少VL调用
spatial_confidence_threshold: float = 0.9

# 启用缓存
enable_cache: bool = True
```

### 处理特殊格式

```python
# 添加自定义题号格式
question_number_patterns = [
    r'^(\d+)[\.、]\s*',        # 默认: 1. 或 1、
    r'^Question\s+(\d+)',      # 自定义: Question 1
]
```

### 一题多图优化

```python
# 调整同行图像识别阈值
same_row_threshold: int = 80  # Y坐标差异阈值（像素）

# 设置最大图片数
max_images_per_question: int = 8
```

## 📝 命令行参数

```bash
python main_extraction.py \
  --result-data <PP-OCR-VL结果路径> \
  --image <原始图像路径> \
  --output <输出JSON路径> \
  --api-key <Qwen-VL API密钥> \
  --visualize  # 生成可视化结果
```

## 🐛 常见问题

### Q: 提示"未设置API密钥"

```bash
# 确认环境变量
echo $DASHSCOPE_API_KEY  # Linux/Mac
echo %DASHSCOPE_API_KEY%  # Windows

# 或直接在命令中指定
python main_extraction.py --api-key "your-key"
```

### Q: 拆分不准确

检查题号格式，可能需要添加自定义模式到配置文件。

### Q: 图像关联错误

调整空间距离阈值：
```python
max_vertical_distance: int = 400  # 增大垂直距离
max_horizontal_distance: int = 300  # 增大水平距离
```

## 📚 进阶使用

查看完整文档：
- [README_EXTRACTION.md](README_EXTRACTION.md) - 完整使用说明
- [docs/题目分割优化方案.md](docs/题目分割优化方案.md) - 技术方案

## 💡 最佳实践

1. **先测试后应用**：使用 `test_extraction.py` 验证功能
2. **检查原始数据**：确保 `result_data.json` 格式正确
3. **查看可视化**：通过可视化图片检查提取结果
4. **调整配置**：根据实际试卷调整参数
5. **启用缓存**：处理大量试卷时启用VL结果缓存

## 🎯 下一步

- 尝试自己的试卷图像
- 根据结果调整配置参数
- 查看详细技术文档了解原理
- 集成到自己的工作流程

祝使用愉快！🎉
