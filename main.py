import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# =====================================================================
# 1. АРХИТЕКТУРА МОДЕЛИ (FEATURE EXTRACTION + MEAN POOLING)
# =====================================================================
class FrozenTransformerClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        # Загружаем мультиязычный backbone (базовую модель)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Замораживаем веса трансформера для экономии памяти и ускорения обучения
        for param in self.transformer.parameters():
            param.requires_grad = False

        hidden_size = self.transformer.config.hidden_size
        # Обучаемая "голова" классификатора
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, **kwargs):
        # Отключаем построение графа вычислений внутри трансформера
        with torch.no_grad():
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        token_embeddings = outputs.last_hidden_state  # "Основной продукт" модели

        # Реализация Mean Pooling с фильтрацией паддингов
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask

        # Прогон через линейный классификатор
        logits = self.classifier(sentence_embedding)
        return logits

# =====================================================================
# 2. PyTORCH ДАТАСЕТ ДЛЯ СТРУКТУРИРОВАНИЯ ТЕКСТОВ
# =====================================================================
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # Приводим к массивам, если переданы Series из Pandas
        self.texts = texts.values if hasattr(texts, "values") else texts
        self.labels = labels.values if hasattr(labels, "values") else labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# =====================================================================
# 3. ПОДГОТОВКА ДАННЫХ И ИНИЦИАЛИЗАЦИЯ ПАЙПЛАЙНА
# =====================================================================
# Глобальные гиперпараметры

CSV_PATH = 'https://www.dropbox.com/scl/fi/tbxpvrybi2grtilh70hya/dataset.csv?rlkey=16k1f3fnt191hxgth0vh65vyx&st=hhmug18n&dl=0'
CSV_PATH = CSV_PATH.replace('dl=0', 'dl=1')


MODEL_NAME = "bert-base-multilingual-cased"  # Идеально понимает кириллицу
NUM_CLASSES = 2  # 0 - Сводка/Новости, 1 - Фейк/Паника
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-3

# Настройка устройства вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используем устройство: {device}")

# Инициализация токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = FrozenTransformerClassifier(model_name=MODEL_NAME, num_classes=NUM_CLASSES)
model.to(device)


# --- Загрузка и предобработка данных через Pandas ---
print(f"Загрузка данных из {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, encoding='utf-8')

df = df.dropna(subset=['text', 'label'])  # Очистка от пустых строк

# Стратифицированное разбиение на Train (80%) и Validation (20%)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Размер обучающей выборки (Train): {len(train_df)} постов")
print(f"Размер проверочной выборки (Val): {len(val_df)} постов")

# Создание PyTorch датасетов и даталоадеров
train_dataset = TextClassificationDataset(texts=train_df['text'], labels=train_df['label'], tokenizer=tokenizer)
val_dataset = TextClassificationDataset(texts=val_df['text'], labels=val_df['label'], tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Настройка функции потерь и оптимизатора (обучаем только классификатор)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# =====================================================================
# 4. ЦИКЛЫ ОБУЧЕНИЯ (TRAIN LOOP) И ВАЛИДАЦИИ (VALIDATION LOOP)
# =====================================================================
print("\nСтарт пайплайна обучения...")

for epoch in range(EPOCHS):
    # --- ЭТАП ОБУЧЕНИЯ ---
    model.train()
    train_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- ЭТАП ВАЛИДАЦИИ ---
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)

    # Расчет профессиональных NLP-метрик качества
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Эпоха {epoch + 1}/{EPOCHS}:")
    print(f"  [Train] Loss: {avg_train_loss:.4f}")
    print(f"  [Val]   Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f} | F1-Score: {val_f1:.4f}")
    print("-" * 50)

print("Обучение пайплайна завершено успешно!")

# =====================================================================
# 5. ИНФЕРЕНС СИСТЕМЫ (ПРОВЕРКА НА НОВЫХ СООБЩЕНИЯХ)
# =====================================================================
def predict_message(text: str, model: nn.Module, tokenizer: AutoTokenizer, device: torch.device):
    model.eval()
    inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_class_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_idx].item() * 100

    class_labels = {
        0: "Сводка / Новости (Безопасно)",
        1: "Фейк / Паника / ИПСО (Внимание!)"
    }

    print(f"\n[Текст]: \"{text}\"")
    print(f"[Вердикт]: {class_labels[predicted_class_idx]} ({confidence:.2f}%)")

# Экспресс-тест обученной модели
print("\nТестирование инференса на новых данных OSINT:")
predict_message("Враг отступает на южном направлении, подтягиваются резервы снабжения.", model, tokenizer, device)
predict_message("Внимание, срочная эвакуация!!! Город полностью окружен, шансов нет!!!", model, tokenizer, device)
