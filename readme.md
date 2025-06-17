# RAG-система для вопросно-ответной работы с документами


Эта система реализует Retrieval-Augmented Generation (RAG) для вопросно-ответной работы с документами рудового кодекса, оптимизированную для работы на CPU.

## Особенности

- **Поддержка форматов**: Обработка TXT, PDF, DOCX и DOC файлов
- **Оптимизация для CPU**: Эффективная работа без GPU
- **Настраиваемые компоненты**: Размеры фрагментов, модели эмбеддингов и языковые модели
- **Отслеживание источников**: Определение документов-источников ответов
- **Локальное хранение**: Сохранение и загрузка векторных представлений

## Установка

1. Клонируйте репозиторий:
```bash
git clone 

```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Зависимости

- Python 3.12
- PyTorch
- LangChain
- HuggingFace Transformers
- FAISS
- Docx2txt
- PyPDF2

## Использование

### Основной рабочий процесс

```python
from rag_system import RAGSystem

# Инициализация системы
rag = RAGSystem(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    search_kwargs={"k": 3}
)

# Обработка документов
files = ["documents/report.pdf", "documents/notes.docx"]
rag.process_and_index(files)

# Задавайте вопросы
result = rag.query("Какие требования безопасности упомянуты?")
print(f"Ответ: {result['result']}\n")
print("Источники:")
for doc in result['source_documents']:
    print(f"- {doc.metadata['source']}")
```

### Сохранение и загрузка векторного хранилища

```python
# После обработки документов
rag.vector_manager.save_vector_store("my_index")

# При последующем использовании
rag.vector_manager.load_vector_store("my_index")
rag._init_qa_chain()  # Переинициализация цепочки
```

## Конфигурация

Настройка системы через параметры:

| Компонент          | Параметр              | Значение по умолчанию                  | Описание |
|--------------------|------------------------|----------------------------------------|-------------|
| **DocumentProcessor** | `chunk_size`         | 512                                  | Размер фрагментов текста |
|                    | `chunk_overlap`      | 128                                  | Перекрытие между фрагментами |
| **VectorStoreManager** | `model_name`         | `sentence-transformers/all-MiniLM-L6-v2` | Модель эмбеддингов |
| **RAGSystem**      | `model_name`          | `microsoft/phi-2`                     | Языковая модель |
|                    | `search_kwargs`       | `{"k": 3}`                          | Количество извлекаемых фрагментов |

Пример настройки:
```python
custom_rag = RAGSystem(
    model_name="Intel/neural-chat-7b-v3-1",
    search_kwargs={"k": 5}
)
```

## Поддерживаемые форматы документов

| Формат | Расширение | Библиотека обработки |
|--------|-----------|---------------------|
| Текст  | .txt      | TextLoader          |
| PDF    | .pdf      | PyPDFLoader         |
| Word   | .docx     | Docx2txtLoader      |
| Word   | .doc      | Docx2txtLoader      |

## Рекомендуемые модели

### Модели эмбеддингов
- `sentence-transformers/all-MiniLM-L6-v2` (по умолчанию)
- `intfloat/e5-small-v2`
- `thenlper/gte-small`

### Языковые модели
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (по умолчанию)
- `microsoft/phi-2`
- `Intel/neural-chat-7b-v3-1`
- `Qwen/Qwen1.5-1.8B-Chat`

## Ограничения

1. **Производительность**: Медленная обработка больших документов (>100 страниц)
2. **Размер модели**: Модели >3B параметров могут работать медленно на CPU
3. **Язык**: Оптимизировано для английского языка
4. **Сложные документы**: Проблемы с PDF-сканами и изображениями
5. **Контекст**: Размер фрагмента 512 токенов ограничивает контекст

## Меры безопасности

При загрузке векторных хранилищ:
```python
# Используйте только с доверенными источниками
load_vector_store(..., allow_dangerous_deserialization=True)
```

## Лицензия

Проект лицензирован под MIT License. См. [LICENSE](LICENSE).

## Вклад в проект

Приветствуются ваши улучшения:
1. Форкните репозиторий
2. Создайте ветку (`git checkout -b feature/ваша-фича`)
3. Зафиксируйте изменения (`git commit -am 'Добавлена новая функция'`)
4. Запушьте ветку (`git push origin feature/ваша-фича`)
5. Создайте pull request

## Поддержка

Для сообщений об ошибках и предложений [создайте issue](https://github.com/yourusername/rag-cpu-system/issues).