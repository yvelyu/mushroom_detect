import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

from app_config import MODEL_DIR, UPLOAD_DIR, ALLOWED_EXTENSIONS, CONFIDENCE_THRESHOLD
from normalization import get_mushroom_info

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

# Загрузка модели и процессора
print("Загрузка модели...")
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
model.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="Файл не выбран")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="Название файла пустое")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Обработка изображения
            image = Image.open(filepath).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            
            # Предсказание
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Получение результатов
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probs, dim=-1)
            
            confidence_val = confidence.item()
            
            if confidence_val < CONFIDENCE_THRESHOLD:
                return render_template('index.html', 
                                     filename=filename,
                                     info={"name_ru": "Не удалось точно определить", "status": "Будьте осторожны"},
                                     confidence=round(confidence_val * 100, 2))

            # Получение метки (обработка случая, когда ключи могут быть как строками, так и числами)
            predicted_idx = predicted_class_idx.item()
            id2label = model.config.id2label
            label = id2label.get(predicted_idx) or id2label.get(str(predicted_idx))
            
            if not label:
                return render_template('index.html', error=f"Мета не найдена для индекса {predicted_idx}")

            info = get_mushroom_info(label)
            
            return render_template('index.html', 
                                 filename=filename, 
                                 info=info, 
                                 confidence=round(confidence_val * 100, 2))
        
        except Exception as e:
            return render_template('index.html', error=f"Ошибка обработки: {str(e)}")

    return render_template('index.html', error="Недопустимый формат файла")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
