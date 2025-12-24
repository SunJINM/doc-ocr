from paddleocr import PPStructureV3
from PIL import Image

init_params = {
    'use_doc_orientation_classify': True,
    'use_doc_unwarping': True,
    'use_textline_orientation': True,
    "use_table_recognition": True,
    'lang': 'ch',
    'device': 'cpu'
}

pipeline = PPStructureV3(**init_params)

image_path = r"D:\WorkProjects\doc-ocr\data\shuxue\1.png"

# 预处理图像，调整尺寸以符合限制
img = Image.open(image_path)
max_size = 4000
if max(img.size) > max_size:
    ratio = max_size / max(img.size)
    new_size = tuple(int(dim * ratio) for dim in img.size)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    # 保存调整后的图像
    processed_path = r"D:\WorkProjects\doc-ocr\data\shuxue\1_resized.png"
    img.save(processed_path)
    image = processed_path
else:
    image = image_path

output = pipeline.predict(input=image)
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output1") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output1") ## 保存当前图像的markdown格式的结果
