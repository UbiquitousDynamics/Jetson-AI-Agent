import os
import pickle
import logging
import hashlib
from typing import Dict, List

class DocumentProcessor:
    """
    Suddivide i documenti di testo in chunk e gestisce la cache per ottimizzare le prestazioni.
    """
    def __init__(self, chars_per_chunk: int = 5000, overlap: int = 200, 
                 cache_path: str = 'chunks.pkl', cache_dir: str = './cache'):
        """
        Inizializza il processore di documenti.
        
        Args:
            chars_per_chunk: Numero di caratteri per chunk
            overlap: Sovrapposizione in caratteri tra chunk consecutivi
            cache_path: Nome del file di cache
            cache_dir: Directory per i file di cache
        """
        self.chars_per_chunk = chars_per_chunk
        self.overlap = min(overlap, chars_per_chunk // 2)  # Evita overlap eccessivi
        
        # Crea la directory di cache se non esiste
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.cache_path = os.path.join(cache_dir, cache_path)
        logging.info(f"DocumentProcessor inizializzato con chunk di {chars_per_chunk} caratteri e overlap di {self.overlap}")

    def _calculate_documents_hash(self, documents: Dict[str, str]) -> str:
        """
        Calcola un hash dai documenti per verificare se sono cambiati.
        
        Args:
            documents: Dizionario di documenti (id -> testo)
            
        Returns:
            Hash dei documenti come stringa
        """
        content = ""
        for doc_id, text in sorted(documents.items()):
            content += f"{doc_id}:{len(text)};"
            content += text[:100] if text else ""  # Usa solo l'inizio del testo per velocità
            
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def chunk_document(self, text: str) -> List[str]:
        """
        Suddivide un singolo documento in chunk.
        
        Args:
            text: Testo del documento
            
        Returns:
            Lista di chunk
        """
        chunks = []
        if not text:
            return chunks
            
        pos = 0
        text_len = len(text)
        
        while pos < text_len:
            chunk_end = min(pos + self.chars_per_chunk, text_len)
            
            # Se non siamo alla fine e non siamo a inizio documento, cerchiamo un punto di rottura migliore
            if chunk_end < text_len and pos > 0:
                # Cerca un punto, una fine paragrafo o uno spazio come punto di rottura
                for break_char in ['. ', '.\n', '\n\n', ' ']:
                    # Cerca il carattere di rottura nella parte finale del chunk
                    search_area = text[chunk_end-50:chunk_end]
                    if break_char in search_area:
                        break_pos = search_area.rindex(break_char) + len(break_char)
                        chunk_end = chunk_end - 50 + break_pos
                        break
            
            # Estrai il chunk
            chunk = text[pos:chunk_end].strip()
            if chunk:  # Ignora chunk vuoti
                chunks.append(chunk)
            
            # Sposta la posizione per il prossimo chunk, considerando l'overlap
            pos = chunk_end - self.overlap if chunk_end < text_len else chunk_end
            
        return chunks

    def load_or_process(self, documents: Dict[str, str], force_reprocess: bool = False) -> List[str]:
        """
        Carica i chunk dalla cache se esistono, altrimenti li elabora e li salva.
        
        Args:
            documents: Dizionario di documenti (id -> testo)
            force_reprocess: Se True, forza la rielaborazione anche se la cache esiste
            
        Returns:
            Lista di chunk elaborati
        """
        documents_hash = self._calculate_documents_hash(documents)
        cache_info_path = f"{self.cache_path}.info"
        
        # 1) Tenta di caricare dalla cache se non è forzata la rielaborazione
        if not force_reprocess and os.path.exists(self.cache_path) and os.path.exists(cache_info_path):
            try:
                # Carica l'hash della cache precedente
                with open(cache_info_path, 'r', encoding='utf-8') as f:
                    cached_hash = f.read().strip()
                    
                # Se l'hash corrisponde, carica i chunk dalla cache
                if cached_hash == documents_hash:
                    with open(self.cache_path, 'rb') as f:
                        chunks = pickle.load(f)
                        logging.info(f"Caricati {len(chunks)} chunk dalla cache")
                        return chunks
            except Exception as e:
                logging.warning(f"Errore nel caricamento della cache: {e}")
        
        # 2) Elabora i documenti in chunk
        chunks = []
        for doc_id, text in documents.items():
            if not text:
                logging.warning(f"Documento vuoto con ID {doc_id}")
                continue
                
            doc_chunks = self.chunk_document(text)
            
            # Aggiungi prefisso ai chunk per tracciare la provenienza
            doc_chunks = [f"[Doc: {doc_id}] {chunk}" for chunk in doc_chunks]
            chunks.extend(doc_chunks)
        
        if not chunks:
            logging.warning("Nessun chunk generato dai documenti forniti")
            return []
            
        # 3) Salva i chunk in cache
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(chunks, f)
                
            # Salva anche l'hash per verifiche future
            with open(cache_info_path, 'w', encoding='utf-8') as f:
                f.write(documents_hash)
                
            logging.info(f"Salvati {len(chunks)} chunk in cache")
        except Exception as e:
            logging.warning(f"Errore nel salvataggio della cache: {e}")
        
        return chunks