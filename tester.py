import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from sentencepiece import SentencePieceProcessor
from typing import List
import textdistance
from typing import List

def process_text(input_text: str) -> str:
    spe_path = "sp.model"  # Путь к файлу SentencePieceProcessor
    tokenizer: SentencePieceProcessor = SentencePieceProcessor(spe_path)

    # Загрузка ONNX модели
    onnx_path = "model.onnx"  # Путь к файлу ONNX модели
    ort_session: ort.InferenceSession = ort.InferenceSession(onnx_path)

    # Загрузка конфигурации модели с метками, параметрами и др.
    config_path = "config.yaml"  # Путь к файлу конфигурации модели
    config = OmegaConf.load(config_path)
    # Возможные метки классификации перед каждым подтокеном
    pre_labels: List[str] = config.pre_labels
    # Возможные метки классификации после каждого подтокена
    post_labels: List[str] = config.post_labels
    # Специальный класс, который означает "ничего не предсказывать"
    null_token = config.get("null_token", "<NULL>")
    # Специальный класс, который означает "все символы в этом подтокене заканчиваются точкой", например, "am" -> "a.m."
    acronym_token = config.get("acronym_token", "<ACRONYM>")
    # Не используется в этом примере, но если ваша последовательность превышает это значение, вам нужно разделить ее на несколько входов
    max_len = config.max_length
    # Для справки: граф не имеет языковой специфики
    languages: List[str] = config.languages

    # Кодирование входного текста, добавление BOS + EOS
    input_ids = [tokenizer.bos_id()] + tokenizer.EncodeAsIds(input_text) + [tokenizer.eos_id()]

    # Создание массива numpy с формой [B, T], как ожидается входом графа.
    input_ids_arr: np.array = np.array([input_ids])

    # Запуск графа, получение результатов для всех аналитических данных
    pre_preds, post_preds, cap_preds, sbd_preds = ort_session.run(None, {"input_ids": input_ids_arr})
    # Убираем измерение пакета и преобразуем в списки
    pre_preds = pre_preds[0].tolist()
    post_preds = post_preds[0].tolist()
    cap_preds = cap_preds[0].tolist()
    sbd_preds = sbd_preds[0].tolist()

    # Обработка текста как ранее
    output_texts: List[str] = []
    current_chars: List[str] = []

    for token_idx in range(1, len(input_ids) - 1):
        token = tokenizer.IdToPiece(input_ids[token_idx])
        if token.startswith("▁") and current_chars:
            current_chars.append(" ")
        # Token-level predictions
        pre_label = pre_labels[pre_preds[token_idx]]
        post_label = post_labels[post_preds[token_idx]]
        # If we predict "pre-punct", insert it before this token
        if pre_label != null_token:
            current_chars.append(pre_label)
        # Iterate over each char. Skip SP's space token,
        char_start = 1 if token.startswith("▁") else 0
        for token_char_idx, char in enumerate(token[char_start:], start=char_start):
            # If this char should be capitalized, apply upper case
            if cap_preds[token_idx][token_char_idx]:
                char = char.upper()
            # Append char
            current_chars.append(char)
            # if this is an acronym, add a period after every char (p.m., a.m., etc.)
            if post_label == acronym_token:
                current_chars.append(".")
        # Maybe this subtoken ends with punctuation
        if post_label != null_token and post_label != acronym_token:
            current_chars.append(post_label)

        # If this token is a sentence boundary, finalize the current sentence and reset
        if sbd_preds[token_idx]:
            output_texts.append("".join(current_chars))
            current_chars.clear()

    # Добавляем последний токен
    output_texts.append("".join(current_chars))

    # Возвращаем обработанный текст
    return "\n".join(output_texts)

def calculate_metrics(predicted: List[str], actual: List[str]):
    tp = fn = fp = 0
    
    for p, a in zip(predicted, actual):
        dist = textdistance.levenshtein(p, a)  
        tp += len(a) - dist
        fn += dist
        fp += len(p) - len(a) + dist
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn) 
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

# Чтение данных из файла
test_data = []
with open("favorable_cases.txt") as f:
    for line in f:
        text1, text2 = line.strip().split("|")
        test_data.append(text1)
        test_data.append(text2)

# Тестирование    
texts = []
predicted = [] 
actual = []

for text in test_data:
    predicted_text = process_text(text)
    texts.append(text)
    predicted.append(predicted_text)
    actual.append(text + " ground truth")
    
precision, recall, f1 = calculate_metrics(predicted, actual) 
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}") 
print(f"F1 score: {f1:.3f}")
