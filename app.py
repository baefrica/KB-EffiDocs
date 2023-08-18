import os
import json
import torch
from flask import Flask, render_template, request
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# KB-ALBERT 모델과 토크나이저 불러오기
kb_albert_model_path = "kb-albert-char-base-v2"
kb_albert_model = AutoModel.from_pretrained(kb_albert_model_path)
tokenizer = AutoTokenizer.from_pretrained(kb_albert_model_path)

# JSON 파일이 들어있는 폴더 경로 지정
json_folder_path = "new_data2222"

# 폴더 내의 모든 JSON 파일 읽기
document_data = []
for filename in os.listdir(json_folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(json_folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            annotations = data["annotations"]
            for annotation in annotations:
                content = annotation["annotation.text"]
                document_data.append((content, file_path))  # 파일 경로도 함께 저장


@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ""
    most_similar_annotations = []
    jpg_filename = ""

    if request.method == 'POST':
        user_input = request.form['user_input']

        # 문서를 KB-ALBERT로 임베딩하여 유사도 계산
        user_input_ids = tokenizer.encode(user_input, return_tensors="pt")
        with torch.no_grad():
            user_embedding = kb_albert_model(user_input_ids)[0].mean(dim=1)

        # 문서의 유사도 계산 및 가장 유사한 문서 찾기
        most_similar_index = None
        max_similarity = -1

        for idx, (doc, file_path) in enumerate(document_data):
            input_ids = tokenizer.encode(doc, return_tensors="pt")
            with torch.no_grad():
                embeddings = kb_albert_model(input_ids)[0].mean(dim=1)
            similarity = cosine_similarity(user_embedding, embeddings).item()

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_index = idx

        if most_similar_index is not None:
            most_similar_json_path = document_data[most_similar_index][1]
            with open(most_similar_json_path, "r", encoding="utf-8") as file:
                most_similar_json_content = json.load(file)

            annotations = most_similar_json_content["annotations"]
            grouped_sentences = []
            y_threshold = 20  # Adjust this threshold based on your needs

            current_sentence = []
            prev_y = None

            for annotation in annotations:
                annotation_text = annotation["annotation.text"]
                bbox = annotation["annotation.bbox"]

                if prev_y is None or abs(bbox[1] - prev_y) <= y_threshold:
                    current_sentence.append(annotation_text)
                else:
                    grouped_sentences.append(" ".join(current_sentence))
                    current_sentence = [annotation_text]

                prev_y = bbox[1]

            if current_sentence:
                grouped_sentences.append(" ".join(current_sentence))

            most_similar_annotations.extend(grouped_sentences)

            json_filename = os.path.basename(most_similar_json_path)
            jpg_filename = os.path.splitext(json_filename)[0] + ".jpg"

    return render_template('index.html', user_input=user_input, most_similar_annotations=most_similar_annotations,
                           jpg_filename=jpg_filename)


if __name__ == '__main__':
    app.run(debug=True)
