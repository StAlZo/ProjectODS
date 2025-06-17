import os
from pathlib import Path

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import torch


class DocumentProcessor:
    """Класс для обработки документов разных форматов"""

    def __init__(self, chunk_size=512, chunk_overlap=128):
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )

        self.loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader
        }

    def load_document(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        if ext not in self.loaders:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")
        loader = self.loaders[ext](file_path)
        return loader.load()

    def process_documents(self, file_paths: list):
        all_docs = []
        for path in file_paths:
            docs = self.load_document(path)
            all_docs.extend(docs)
        return self.text_splitter.split_documents(all_docs)


class VectorStoreManager:
    """Управление векторными хранилищами"""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = None

    def create_vector_store(self, documents):
        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )
        return self.vector_store

    def save_vector_store(self, path="faiss_index"):
        if self.vector_store:
            self.vector_store.save_local(path)

    def load_vector_store(self, path="faiss_index"):
        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vector_store


class RAGSystem:
    """RAG система для работы на CPU"""

    def __init__(self,
                 model_name="microsoft/phi-2",
                 search_kwargs={"k": 3}):

        self.processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager()
        self.qa_chain = None
        self.search_kwargs = search_kwargs

        # Инициализация модели для CPU
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=None  # Отключаем автоматическое распределение
        )
        self.model.to("cpu")  # Явно перемещаем на CPU

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            repetition_penalty=1.15
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipe)  # Новый импорт

    def process_and_index(self, file_paths: list):
        # Проверяем существование файлов перед обработкой
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл не найден: {path}")

        documents = self.processor.process_documents(file_paths)
        self.vector_manager.create_vector_store(documents)
        self._init_qa_chain()

    def _init_qa_chain(self):
        if not self.vector_manager.vector_store:
            raise ValueError("Векторное хранилище не инициализировано")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_manager.vector_store.as_retriever(
                search_kwargs=self.search_kwargs
            ),
            return_source_documents=True
        )

    def query(self, question: str):
        if not self.qa_chain:
            raise ValueError("QA цепочка не инициализирована")

        return self.qa_chain({"query": question})


# Пример использования
if __name__ == "__main__":
    # Инициализация системы
    rag = RAGSystem(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        search_kwargs={"k": 1}
    )

    # Обработка файлов
    files = ["documents/Трудовой кодекс Российской Федерации от 30.12.2001 N 197-ФЗ.docx", ]
    rag.process_and_index(files)

    # Запрос
    result = rag.query("Что говорится о безопасности в документах?")

    print(f"Ответ: {result['result']}\n")
    print("Источники:")
    for doc in result['source_documents']:
        print(f"- {Path(doc.metadata['source']).name}")