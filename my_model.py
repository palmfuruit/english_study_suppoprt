import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_labels(texts, threshold=0.5):
    # モデルとトークナイザーのロード
    print('modelとtokenizerのロード ---start---')
    model = AutoModelForSequenceClassification.from_pretrained('./model')
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
    print('modelとtokenizerのロード ---end---')

    model = model.to('cpu')

    # テキストをトークナイズ
    encodings = tokenizer(
        texts,
        max_length=32,  # ここで適切な最大長を設定
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'].to('cpu')
    attention_mask = encodings['attention_mask'].to('cpu')

    # モデルで推論を行う
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    logits = outputs.logits

    # Sigmoid関数を適用して確率を計算
    sigmoid = nn.Sigmoid()
    probs = sigmoid(logits)

    # 各入力データに対して予測ラベルを計算
    preds = (probs >= threshold).float().cpu().numpy()

    # 予測結果のリストを返す
    return preds.tolist()


# # 使用例
# sentences = [
#     "This is the first test sentence.",
#     "The book was read by the entire class.",
#     "She is taller than her brother.",
#     "The teacher made the students write an essay.",
#     "If I were you, I would apologize immediately.",
#     "The movie that we watched yesterday was amazing.",
#     "This is the place where we first met."
# ]
# predicted_labels = predict_labels(sentences, models, vectorizer)

# # 予測結果の表示
# for i, labels in enumerate(predicted_labels):
#     print(f"Sentence {i+1}: {sentences[i]}")
#     print(f"Predicted Labels: {labels}")
#     print()