from paddleocr import PaddleOCR

text_detection_model_dir = "D:\models"
text_recognition_model_dir = "D:\models"


init_params = {
    'use_doc_orientation_classify': True,
    'use_doc_unwarping': True,
    'use_textline_orientation': True,
    'lang': 'ch',
    'device': 'cpu'
}

ocr_engine = PaddleOCR(**init_params)

image = r"D:\WorkProjects\doc-ocr\data\input\1.jpg"
res = ocr_engine.predict(input=image, return_word_box=True)

for res in res:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
