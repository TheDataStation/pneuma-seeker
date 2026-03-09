import shutil
import bm25s
import os

from Stemmer import Stemmer

from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    Knowledge,
    RetrieverType,
    Text,
)
from pneuma_seeker.services.core.ir_system.retriever.interface import AbstractRetriever


class DocumentDB(AbstractRetriever):
    """Represents a domain knowledge retriever."""

    def __init__(self, user_id: str, chat_id: str, config, db_api, language_model_api):
        super().__init__(user_id, chat_id, config, db_api, language_model_api)
        self.local_retriever = None
        self.global_retriever = None

        curr_file_path = os.path.dirname(os.path.abspath(__file__))
        self.LOCAL_INDEX_PATH = os.path.join(
            curr_file_path, "indices", "kb", "local"
        )
        self.GLOBAL_INDEX_PATH = os.path.join(
            curr_file_path, "indices", "kb", "global"
        )
        self.stemmer = Stemmer("english")

    @property
    def retriever_type(self) -> RetrieverType:
        """
        Defines the type of the retriever.
        """
        return RetrieverType.DOCUMENT_DB

    def load(self):
        """
        Loads the retriever, including its dependencies (e.g., its model).
        """
        if self.local_retriever is None:
            try:
                self.local_retriever = bm25s.BM25.load(
                    self.LOCAL_INDEX_PATH, load_corpus=True
                )
            except:
                pass
        if self.global_retriever is None:
            try:
                self.global_retriever = bm25s.BM25.load(
                    self.GLOBAL_INDEX_PATH, load_corpus=True
                )
            except:
                pass

    def retrieve(
        self,
        query: str,
        k: int,
        sample_only: bool,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        """
        Retrieves a list of documents given a query.
        """
        self.load()
        if self.local_retriever is None and self.global_retriever is None:
            print("Both the local and global retrievers have not been initialized.")
            return []
        retrieval_results: list[AbstractDocument] = []
        if self.local_retriever is not None:
            retrieval_results.extend(self.__actual_retrieve(query, self.local_retriever, k))
        if self.global_retriever is not None:
            retrieval_results.extend(
                self.__actual_retrieve(query, self.global_retriever, k)
            )

        self.local_retriever = None
        self.global_retriever = None
        return retrieval_results

    def __actual_retrieve(
        self, query: str, retriever: bm25s.BM25, k: int
    ) -> list[AbstractDocument]:
        if retriever.corpus is None:
            raise ValueError("Both the retriever or its corpus cannot be None.")
        max_k = min(len(retriever.corpus), k)
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
        results, _ = retriever.retrieve(query_tokens, k=max_k, show_progress=False)
        retrieval_results: list[AbstractDocument] = []
        for result in results[0]:
            retrieval_results.append(
                Text(
                    doc_id=result["metadata"]["doc_id"],
                    retriever_type=RetrieverType.DOCUMENT_DB,
                    content=result["text"],
                    metadata={
                        "type": result["metadata"]["type"],
                        "user": result["metadata"]["user"],
                    },
                )
            )
        return retrieval_results

    def index(self, documents: list[AbstractDocument]):
        """
        Indexes a list of documents to the retriever.
        """
        # Future-TODO: Handle possibility of conflicts (LLM required here!)
        corpus_json_local: list[dict] = []
        corpus_json_global: list[dict] = []

        new_global_document = False
        new_local_document = False
        
        # Get existing documents first
        if os.path.exists(self.LOCAL_INDEX_PATH):
            retriever = bm25s.BM25.load(self.LOCAL_INDEX_PATH, load_corpus=True)
            if retriever.corpus is not None:
                corpus_json_local.extend(retriever.corpus)
        
        if os.path.exists(self.GLOBAL_INDEX_PATH):
            retriever = bm25s.BM25.load(self.GLOBAL_INDEX_PATH, load_corpus=True)
            if retriever.corpus is not None:
                corpus_json_global.extend(retriever.corpus)

        # Calculate the next available ID
        existing_ids = set()
        for corpus in [corpus_json_local, corpus_json_global]:
            for doc in corpus:
                doc_id = int(doc["metadata"]["doc_id"].split("_")[1])
                existing_ids.add(doc_id)
        next_id = max(existing_ids) + 1 if existing_ids else 0

        # Process new documents
        for document in documents:
            if not isinstance(document, Knowledge):
                raise ValueError("All documents must be of type Knowledge.")
            
            doc_entry = {
                "text": document.content,
                "metadata": {
                    "doc_id": f"kb_{next_id}",
                    "type": document.metadata["type"],
                    "user": document.metadata["user"],
                }
            }
            next_id += 1

            if document.metadata["type"] == "local":
                new_local_document = True
                corpus_json_local.append(doc_entry)
            elif document.metadata["type"] == "global":
                new_global_document = True
                corpus_json_global.append(doc_entry)

        # Save indices if we have new documents
        if new_global_document:
            if os.path.exists(self.GLOBAL_INDEX_PATH):
                shutil.rmtree(self.GLOBAL_INDEX_PATH)
            corpus_text = [doc["text"] for doc in corpus_json_global]
            corpus_tokens = bm25s.tokenize(
                corpus_text, stopwords="en", stemmer=self.stemmer, show_progress=False
            )
            retriever = bm25s.BM25(corpus=corpus_json_global)
            retriever.index(corpus_tokens, show_progress=True)
            retriever.save(self.GLOBAL_INDEX_PATH, corpus=corpus_json_global)
        
        if new_local_document:
            if os.path.exists(self.LOCAL_INDEX_PATH):
                shutil.rmtree(self.LOCAL_INDEX_PATH)
            corpus_text = [doc["text"] for doc in corpus_json_local]
            corpus_tokens = bm25s.tokenize(
                corpus_text, stopwords="en", stemmer=self.stemmer, show_progress=False
            )
            retriever = bm25s.BM25(corpus=corpus_json_local)
            retriever.index(corpus_tokens, show_progress=True)
            retriever.save(self.LOCAL_INDEX_PATH, corpus=corpus_json_local)
