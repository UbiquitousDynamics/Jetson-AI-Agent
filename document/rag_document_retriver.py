import os
import logging
from typing import Dict, List, Tuple, Optional
import time
from document.document_processor import DocumentProcessor
from document.rag_system import RagSystem

class RagDocumentRetriever:
    """
    Sistema RAG ottimizzato con:
      - chunking intelligente con sovrapposizione
      - cache di chunk ed embeddings basata su hash
      - caricamento efficiente da cache
      - gestione degli errori robusta
    """
    def __init__(
        self,
        documents: Dict[str, str],
        model_name: str = 'BAAI/bge-base-en-v1.5',
        cache_dir: str = './rag_cache',
        chunks_cache: str = 'chunks.pkl',
        embeddings_path: str = 'embeddings.npy',
        sentences_path: str = 'sentences.txt',
        chars_per_chunk: int = 1000,
        overlap: int = 100,
        max_chunks: int = 300,
        use_fp16: bool = True
    ):
        """
        Inizializza il sistema RAG per il recupero di documenti.
        
        Args:
            documents: Dizionario di documenti (id -> testo)
            model_name: Nome del modello di embedding
            cache_dir: Directory per i file di cache
            chunks_cache: Nome del file di cache dei chunk
            embeddings_path: Nome del file degli embedding
            sentences_path: Nome del file dei testi originali
            chars_per_chunk: Numero di caratteri per chunk
            overlap: Sovrapposizione tra chunk consecutivi
            max_chunks: Numero massimo di chunk da processare
            use_fp16: Se utilizzare fp16 per accelerare il calcolo
        """
        self.rag_system = None
        self.cache_dir = cache_dir
        
        # Assicurati che la directory di cache esista
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Percorsi completi per i file di cache
        #chunks_cache_path = os.path.join(cache_dir, chunks_cache)
        embeddings_path = os.path.join(cache_dir, embeddings_path)
        sentences_path = os.path.join(cache_dir, sentences_path)
        
        start_time = time.time()
        logging.info("[RAG] Inizializzazione del retriever...")
        
        # 1) Carica o calcola i chunk
        processor = DocumentProcessor(
            chars_per_chunk=chars_per_chunk, 
            overlap=overlap,
            cache_path=chunks_cache,
            cache_dir=cache_dir
        )
        
        chunks = processor.load_or_process(documents)
        if not chunks:
            logging.warning("[RAG] Nessun chunk generato. Impossibile procedere.")
            return
            
        # Limita il numero di chunk per gestione memoria/calcolo
        if len(chunks) > max_chunks:
            logging.warning(f"[RAG] Limitando da {len(chunks)} a {max_chunks} chunk")
            chunks = chunks[:max_chunks]

        # 2) Inizializza il sistema RAG
        try:
            self.rag_system = RagSystem(model_name=model_name, use_fp16=use_fp16)
        except Exception as e:
            logging.error(f"[RAG] Errore nell'inizializzazione del modello: {e}")
            return
            
        # 3) Carica gli embeddings esistenti o creane di nuovi
        try:
            self.rag_system.index_database(
                chunks, 
                save_path=embeddings_path, 
                text_data_path=sentences_path
            )
            logging.info(f"[RAG] Sistema pronto con {len(chunks)} chunk")
        except Exception as e:
            logging.error(f"[RAG] Errore nell'indicizzazione del database: {e}")
            
        end_time = time.time()
        logging.info(f"[RAG] Inizializzazione completata in {end_time - start_time:.2f} secondi")

    def retrieve(self, query: str, top_k: int = 1, min_score: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Recupera i chunk più rilevanti per una query.
        
        Args:
            query: Query di ricerca
            top_k: Numero di risultati da restituire
            min_score: Punteggio minimo di similarità per i risultati
            
        Returns:
            Lista di tuple (id_chunk, testo, punteggio)
        """
        if not self.rag_system or self.rag_system.embedding_matrix is None:
            logging.warning("[RAG] Retrieval non disponibile. Sistema non inizializzato correttamente.")
            return []
            
        try:
            start_time = time.time()
            # Ottieni i risultati dalla ricerca
            results = self.rag_system.search(query, top_k=top_k)
            
            # Filtra per punteggio minimo e formatta i risultati
            filtered_results = []
            for idx, score in results:
                if score >= min_score:
                    # Estrai l'ID del documento dal testo se presente
                    text = self.rag_system.data[idx]
                    doc_id = "unknown"
                    
                    # Cerca il formato "[Doc: id_doc]" all'inizio del testo
                    if text.startswith("[Doc:"):
                        end_bracket = text.find("]")
                        if end_bracket > 0:
                            doc_id = text[6:end_bracket].strip()
                            text = text[end_bracket+1:].strip()
                    
                    filtered_results.append((doc_id, text, score))
                    
            end_time = time.time()
            logging.info(f"[RAG] Retrieval completato in {end_time - start_time:.4f} secondi")
            logging.info(f"[RAG] Trovati {len(filtered_results)} risultati con score >= {min_score}")
            
            return filtered_results
        except Exception as e:
            logging.error(f"[RAG] Errore durante il retrieval: {e}")
            return []
            
    def visualize_query_space(self, query: str, perplexity: int = 5, 
                             save_path: Optional[str] = None) -> None:
        """
        Visualizza lo spazio semantico con la query.
        
        Args:
            query: Query da visualizzare
            perplexity: Parametro di perplexity per t-SNE
            save_path: Se specificato, salva la figura in questo percorso
        """
        if not self.rag_system or self.rag_system.embedding_matrix is None:
            logging.warning("[RAG] Visualizzazione non disponibile. Sistema non inizializzato.")
            return
            
        try:
            if save_path and not save_path.startswith(self.cache_dir):
                save_path = os.path.join(self.cache_dir, save_path)
                
            self.rag_system.visualize_space_query(
                query=query, 
                perplexity=perplexity,
                save_path=save_path
            )
        except Exception as e:
            logging.error(f"[RAG] Errore durante la visualizzazione: {e}")