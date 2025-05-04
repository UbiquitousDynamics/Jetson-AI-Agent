import logging
import os
import threading
import queue
import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)

import config

logging.getLogger("comtypes").setLevel(logging.WARNING)

class APIClient:
    def __init__(
        self,
        model_talk: str = config.MODEL_TALK,
        model_think: str = config.MODEL_THINK,
        device: int = int(config.DEVICE) if isinstance(config.DEVICE, (str, int, float)) else -1,  # Default to -1 if invalid
        cache_dir: str = ".models_cache",
        revision_talk: str = None,
        revision_think: str = None,
    ):
        """
        Inizializza le pipeline 'talk' e 'think'.
        """
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Caricamento modelli: talk={model_talk}, think={model_think}")

        # Caricamento artefatti
        self.talk_model, self.talk_tokenizer = self._load_artifacts(
            model_talk, cache_dir, revision_talk
        )
        self.think_model, self.think_tokenizer = self._load_artifacts(
            model_think, cache_dir, revision_think
        )

        # Creazione pipeline
        self.talk_pipeline = pipeline(
            "text-generation",
            model=self.talk_model,
            tokenizer=self.talk_tokenizer,
            device=device,
        )
        self.think_pipeline = pipeline(
            "text-generation",
            model=self.think_model,
            tokenizer=self.think_tokenizer,
            device=device,
        )

        # Evento unico per stoppare lo streaming
        self.stop_event = threading.Event()
        self.command_queue = queue.Queue()

    def _load_artifacts(self, name, cache_dir, revision=None):
        logging.info(f"Caricamento modello/tokenizer {name}")
        model = AutoModelForCausalLM.from_pretrained(
            name, cache_dir=cache_dir, revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(
            name, cache_dir=cache_dir, revision=revision
        )
        device_name = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device_name)
        logging.debug(f"{name} caricato su {device_name}")
        return model, tokenizer

    def _listener(self, callback=None):
        """
        Thread che legge i comandi e, se è 'stop', setta stop_event.
        """
        while not self.stop_event.is_set():
            try:
                cmd = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if cmd.lower() in ("stop", "interrompi", "silenzio", "basta"):
                logging.info("Interruzione streaming richiesta")
                self.stop_event.set()
                break
            if callback:
                callback(cmd)

    def talk_direct_tts(
        self,
        message: str,
        tts_callback=None,
        command_callback=None,
    ) -> None:
        """
        Versione semplificata che usa solo TTS senza LLM.
        Divide il testo in frasi e le pronuncia sequenzialmente.
        """
        self.stop_event.clear()
        
        # Avvia il listener per i comandi di interruzione
        listener = threading.Thread(
            target=self._listener,
            args=(command_callback,),
            daemon=True
        )
        listener.start()
    
        # Regex per dividere il testo in frasi
        sentence_pattern = re.compile(r'[.!?]+[\s\n]+|[.!?]+$')
        
        # Dividi il messaggio in frasi
        sentences = sentence_pattern.split(message.strip())
        
        for sentence in sentences:
            if self.stop_event.is_set():
                logging.info("TTS interrotto")
                break
                
            # Pulisci e valida la frase
            sentence = sentence.strip()
            if len(sentence.split()) > 2:  # Pronuncia solo frasi con almeno 3 parole
                if tts_callback:
                    logging.debug(f"TTS in esecuzione per: {sentence}")
                    if tts_callback(sentence) is False:
                        self.stop_event.set()
                        break
                else:
                    print(f"[TTS] {sentence}")
                        
        logging.debug("TTS completato")

    def talk_stream_tts_phrase(
        self,
        message: str,
        context: str = None,
        tts_callback=None,
        command_callback=None,
    ) -> None:
        """
        Streaming TTS frase per frase con interruzione possibile.
        """
        # Prepara prompt con istruzioni sulla lingua e formato
        prompt = self._format_prompt(message, context)
        
        # Log del prompt completo a livello di debug
        logging.debug(f"Prompt completo inviato al modello: {prompt[:100]}...")
        
        inputs = self.talk_tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.talk_model.device) for k, v in inputs.items()}
    
        # Streamer con skip_prompt=True
        streamer = TextIteratorStreamer(
            self.talk_tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        gen_kwargs = {
            "max_new_tokens": 150,
            "streamer": streamer,
            "do_sample": True,
            "temperature": 0.7
        }
    
        # Reset e avvio listener
        self.stop_event.clear()
        listener = threading.Thread(
            target=self._listener, args=(command_callback,), daemon=True
        )
        listener.start()
    
        # Avvio generazione
        gen_thread = threading.Thread(
            target=self.talk_model.generate, kwargs={**inputs, **gen_kwargs}, daemon=True
        )
        gen_thread.start()
    
        # Raccoglimento e gestione dello streaming per frasi complete
        collecting_response = False
        buffer = ""
        sentence_buffer = ""
        
        # Usa una regex per identificare la fine di una frase
        sentence_end_pattern = re.compile(r'[.!?][\s"]')
        
        for token in streamer:
            if self.stop_event.is_set():
                logging.info("Streaming interrotto")
                break
                
            # Filtra istruzioni e contenuti di sistema
            if not collecting_response:
                if token.strip() and not any(x in token.lower() for x in 
                                           ["istruzioni:", "instruction:", "context:", "contesto:", 
                                            "history:", "user:", "risposta:", "prompt:"]):
                    collecting_response = True
                else:
                    continue
                
            if collecting_response:
                buffer += token
                sentence_buffer += token
                
                # Verifica se abbiamo una frase completa
                if sentence_end_pattern.search(sentence_buffer):
                    # Abbiamo una frase completa
                    sentences = sentence_end_pattern.split(sentence_buffer)
                    for i, sentence in enumerate(sentences[:-1]):  # L'ultimo frammento è incompleto
                        full_sentence = sentence.strip() + "."  # Aggiungi il punto che è stato rimosso dallo split
                        
                        # Pronuncia solo se la frase è abbastanza lunga e non contiene elementi di prompt
                        if len(full_sentence.split()) > 3 and not any(x in full_sentence.lower() for x in 
                                                                   ["istruzioni:", "instruction:", "contesto:", 
                                                                    "context:", "history:", "user:", "risposta:", "prompt:"]):
                            if tts_callback:
                                logging.debug(f"Performing TTS for complete sentence: {full_sentence}")
                                cont = tts_callback(full_sentence)
                                if cont is False:
                                    self.stop_event.set()
                                    break
                            else:
                                print(f"[TTS] {full_sentence}")
                    
                    # Mantieni l'ultima parte incompleta
                    sentence_buffer = sentences[-1] if sentences else ""
    
        # Gestione di eventuali frasi residue nel buffer
        if sentence_buffer and collecting_response and not self.stop_event.is_set():
            sentence = sentence_buffer.strip()
            if sentence and len(sentence.split()) > 3 and not any(x in sentence.lower() for x in 
                                                              ["istruzioni:", "instruction:", "contesto:", 
                                                               "context:", "history:", "user:", "risposta:", "prompt:"]):
                if tts_callback:
                    logging.debug(f"Performing TTS for final text: {sentence}")
                    tts_callback(sentence)
                else:
                    print(f"[TTS] {sentence}")
    
        # Assicurati che il thread di generazione sia terminato
        gen_thread.join(timeout=1.0)

    def _format_prompt(self, message, context=None):
        """
        Format the prompt with appropriate system instructions.
        Use a clear delimiter to separate instructions and the expected response.
        """
        system_instruction = """You are Emilia 5, an intelligent solar car. Respond clearly, concisely, and naturally.
    For questions like "How are you" or "Who are you," respond in a conversational and personal manner.
    Your responses must be brief, relevant, and suitable for a vocal conversation.
    
    IMPORTANT: The text after "RESPONSE:" will be the only part spoken aloud. 
    All your responses must start directly with the actual response, without preambles or text explaining what you are doing."""
    
        if context:
            return f"{system_instruction}\n\nContext:\n{context}\n\nPrompt:\n{message}\n\nRISPOSTA:"
        else:
            return f"{system_instruction}\n\nPrompt:\n{message}\n\nRISPOSTA:"

    def think(self, message: str, context: str = None) -> str:
        """
        Genera testo con il modello 'think'.
        """
        prompt = self._format_prompt(message, context)
        inputs = self.think_tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.think_model.device) for k, v in inputs.items()}
        outputs = self.think_model.generate(**inputs, max_new_tokens=150)  # Increased max tokens
        text = self.think_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Estrai solo la risposta, rimuovendo il prompt
        response_text = text[len(prompt):].strip() if text.startswith(prompt) else text
        return response_text

    def add_command(self, command: str):
        self.command_queue.put(command)

    def interrupt_generation(self):
        self.stop_event.set()
        logging.info("Interruzione generazione richiesta")