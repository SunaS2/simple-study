import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import torch

# nltk의 불용어와 표제어 추출 준비
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# CSV 파일 로드
data = pd.read_csv('03. score-difficulty/cefr_leveled_texts.csv')  # 파일명을 실제 파일명으로 변경하세요

# 텍스트 전처리 함수 정의
def preprocess_text(text):
    # 1. 토큰화 및 소문자 변환
    tokens = text.lower().split()
    
    # 2. 불용어 제거
    tokens = [word for word in tokens if word not in stop_words]
    
    # 3. 표제어 추출
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 전처리된 텍스트 반환
    return ' '.join(tokens)

# 데이터에 전처리 적용
data['processed_text'] = data['text'].apply(preprocess_text)

# BERT 임베딩 함수 정의
def get_bert_embeddings(text):
    # 입력 텍스트를 BERT의 입력 형식으로 변환
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 임베딩 결과는 모델의 마지막 은닉층
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 평균 풀링으로 문장 벡터 생성
    return embeddings.squeeze().numpy()

# BERT 임베딩 생성 및 적용
data['bert_embeddings'] = data['processed_text'].apply(get_bert_embeddings)

# 결과 확인
print(data[['text', 'label', 'processed_text', 'bert_embeddings']].head())


import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

# 전처리된 데이터 로드
# data = pd.read_csv('./cefr_leveled_texts.csv')  # 실제 파일명으로 변경하세요
X = np.stack(data['bert_embeddings'].values)   # BERT 임베딩을 사용한 입력 데이터
y = data['label'].values                       # CEFR 레벨 라벨

# 데이터 분할 (80% 학습용, 20% 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


training_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)

# 교차 검증을 통한 성능 평가
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(training_model, X_train, y_train, cv=kfold, scoring='accuracy')

print("K-Fold Cross-Validation Accuracy:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# 최종 학습 및 예측
training_model.fit(X_train, y_train)
y_pred = training_model.predict(X_test)

# 모델 평가
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 영화 대사 난이도 예측 함수
def predict_difficulty(dialogues):
    dialogues_processed = [preprocess_text(dialogue) for dialogue in dialogues]  # 전처리 적용
    embeddings = [get_bert_embeddings(dialogue) for dialogue in dialogues_processed]  # BERT 임베딩 적용
    predictions = training_model.predict(embeddings)
    return predictions

# 예시: 새로운 영화 대사 난이도 예측
new_dialogues = ["In view of this, the Ethiopian Government and other developmental partners have introduced an extensive mechanical and biological watershed conservation schemes in various parts of the country over the last decades particularly after the famine of the 1970s"]
difficulty_predictions = predict_difficulty(new_dialogues)
print("Predicted CEFR Levels:", difficulty_predictions)
