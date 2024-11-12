from collections import Counter
import re

def evaluate_difficulty(text):
    # 어휘 빈도 및 난이도
    word_tokens = re.findall(r'\b\w+\b', text.lower())
    word_count = len(word_tokens)
    unique_words = len(set(word_tokens))
    ttr = unique_words / word_count if word_count != 0 else 0  # 어휘 다양성
    
    # 문장 길이 및 복잡도 (평균 문장 길이 계산)
    sentences = re.split(r'[.!?]', text)
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences if sentence) / len(sentences) if sentences else 0
    
    # 단어 난이도 (예: CEFR 사전을 활용하여 점수 매핑)
    # 예시로 간단히 구현 (실제는 CEFR 어휘 목록 필요)
    basic_vocab = set(["hello", "world", "basic", "text"])  # 예시용 기본 단어
    advanced_vocab = set(["complexity", "elaborate", "advanced"])  # 예시용 고급 단어
    basic_words = sum(1 for word in word_tokens if word in basic_vocab)
    advanced_words = sum(1 for word in word_tokens if word in advanced_vocab)
    vocab_difficulty_score = 50 * (advanced_words / word_count) if word_count != 0 else 0
    
    # 구어체 비율 평가 (예: 대화체 표현 비율 측정)
    informal_phrases = set(["gonna", "wanna", "ain't", "dunno"])
    informal_count = sum(1 for word in word_tokens if word in informal_phrases)
    informal_ratio = informal_count / word_count if word_count != 0 else 0
    
    # 종합 점수 산출 (단순 예시)
    total_score = (ttr * 25) + (avg_sentence_length * 20) + vocab_difficulty_score + (informal_ratio * 10)
    
    # CEFR 등급 매핑
    if total_score <= 20:
        level = "A1"
    elif total_score <= 35:
        level = "A2"
    elif total_score <= 50:
        level = "B1"
    elif total_score <= 70:
        level = "B2"
    elif total_score <= 85:
        level = "C1"
    else:
        level = "C2"
    
    return {
        "total_score": total_score,
        "CEFR_level": level
    }

# 테스트 예시
text_example = "Hello"
result = evaluate_difficulty(text_example)
print("점수:", result["total_score"])
print("CEFR 레벨:", result["CEFR_level"])
