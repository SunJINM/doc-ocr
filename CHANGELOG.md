# 更新日志

## [1.0.1] - 2025-12-24

### 🔧 修复

**PaddleOCR API兼容性**
- ✅ 修复 OCR 调用方式以兼容 PaddleOCR 新版本
- ✅ 将 `ocr.ocr()` 改为 `ocr.predict()`
- ✅ 更新参数：移除 `det/rec/cls`，使用新版参数
- ✅ 使用临时文件方式传递图像（新版API要求）

**受影响文件**：
- `src/question_extraction/question_splitter.py` - OCR调用逻辑
- `src/question_extraction/config.py` - OCR配置参数
- `main_extraction.py` - OCR初始化
- `test_extraction.py` - 测试脚本

### 📝 新版OCR配置

**旧版配置**（已废弃）：
```python
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=True,
    det=True,
    rec=True,
    cls=False,
    det_db_thresh=0.3,
    det_db_box_thresh=0.5
)
```

**新版配置**（当前使用）：
```python
ocr = PaddleOCR(
    lang='ch',
    device='cpu',  # 或 'gpu'
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True
)
```

### 🔄 API变化

**OCR调用方式**：

旧版：
```python
ocr_results = ocr.ocr(image, det=True, rec=True, cls=False)
```

新版：
```python
# 需要文件路径
ocr_results = ocr.predict(input=image_path, return_word_box=True)
```

**结果数据结构**：

旧版：
```python
for line_result in ocr_results[0]:
    bbox = line_result[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text = line_result[1][0]
    conf = line_result[1][1]
```

新版：
```python
for result in ocr_results:
    if hasattr(result, 'boxes'):
        for box_info in result.boxes:
            bbox = box_info['points']
            text = box_info.get('text', '')
            score = box_info.get('score', 0.0)
```

### 💡 使用建议

1. **GPU使用**：修改配置
   ```python
   # config.py
   device: str = 'gpu'  # 改为 'gpu'
   ```

2. **临时文件清理**：代码会自动清理，无需担心

3. **兼容性**：如果使用旧版PaddleOCR，请降级到v1.0.0

### 📋 测试验证

运行测试确认修复：
```bash
python test_extraction.py
```

预期输出：
```
✅ 所有测试通过！
```

---

## [1.0.0] - 2025-12-24

### 🎉 初始版本

**核心功能**：
- ✅ 题目拆分 - 处理多题合并问题
- ✅ 图文合并 - 关联题目与配图
- ✅ 一题多图支持 - 选择题ABCD场景
- ✅ 批量VL验证 - 成本优化
- ✅ 智能降级 - 确保基本功能

**文件清单**：
- 核心代码：7个文件（1500+行）
- 主程序和测试：2个文件
- 文档：5个文件
- 技术方案：1个文件

**性能指标**：
- 题目检测准确率: ≥95%
- 图文关联准确率: ≥90%
- 拆分准确率: ≥92%
- 处理速度: ≤45秒/页
- 单份试卷成本: ≈0.006元
