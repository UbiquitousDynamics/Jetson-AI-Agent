import os
import numpy as np
import logging
from rag_oo_master.document_processor import DocumentProcessor
from rag_oo_master.rag_system import RagSystem

class RagDocumentRetriever:
    """
    RAG ottimizzato con:
      - chunk grandi per ridurre il conteggio;
      - cache di chunk ed embeddings;
      - caricamento in <1s se la cache esiste.
    """
    def __init__(
        self,
        documents: dict,
        model_name: str = 'BAAI/bge-m3',
        chunks_cache: str = 'chunks.pkl',
        embeddings_path: str = 'embeddings.npy',
        sentences_path: str = 'sentences.txt',
        max_chunks: int = 50
    ):
        self.rag_system = None
        # 1) Carica o calcola i chunk
        processor = DocumentProcessor(cache_path=chunks_cache)
        chunks = processor.load_or_process(documents)
        if not chunks:
            logging.warning("[RAG] Nessun chunk generato.")
            return
        # Limita il numero di chunk per RAM/calc
        chunks = chunks[:max_chunks]

        # 2) Cache embeddings
        if os.path.exists(embeddings_path) and os.path.exists(sentences_path):
            try:
                matrix = np.load(embeddings_path)
                with open(sentences_path, 'r', encoding='utf-8') as f:
                    saved = [line.strip() for line in f if line.strip()]
                if len(saved) == matrix.shape[0]:
                    logging.info(f"[RAG] Caricati {len(saved)} embeddings cached.")
                    self.rag_system = RagSystem(model_name=model_name)
                    self.rag_system.embedding_matrix = matrix
                    self.rag_system.data = saved
                    return
            except Exception as e:
                logging.warning(f"[RAG] Cache embeddings fallita: {e}")

        # 3) Crea e salva embeddings una tantum
        self.rag_system = RagSystem(model_name=model_name)
        try:
            embs = self.rag_system.model.encode(chunks)['dense_vecs']
            matrix = np.array(embs)
            self.rag_system.embedding_matrix = matrix
            self.rag_system.data = chunks
            np.save(embeddings_path, matrix)
            with open(sentences_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(chunks))
            logging.info(f"[RAG] Salvati {len(chunks)} embeddings su disco.")
        except Exception as e:
            logging.error(f"[RAG] Errore embedding: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str,str]]:
        if not self.rag_system or self.rag_system.embedding_matrix is None:
            logging.warning("[RAG] Retrieval non disponibile.")
            return []
        results = self.rag_system.search(query, top_k=top_k)
        return [(f"chunk_{i+1}", self.rag_system.data[i]) for i,_ in results]