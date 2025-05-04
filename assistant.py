import logging
import time
from threading import Thread, Lock, Event
import queue
import speech_recognition as sr
from api.api_client_advanced import APIClient
from recognizer.speech_recognizer import SpeechRecognizer
from document.rag_document_retriver import RagDocumentRetriever
import config

if config.OS_ == config.OS.WINDOWS:
    from audio.win_tts import Pyttsx3TTS
    from audio.sound_player import SoundPlayer
else:
    from audio.tts import Pyttsx3TTS
    from audio.os_sound_player import SoundPlayer

class VoiceAssistant:
    def __init__(self, documents: dict = None):
        # Inizializzazione componenti principali
        self.api = APIClient()
        self.tts = Pyttsx3TTS()
        self.sound = SoundPlayer()
        self.rec_main = SpeechRecognizer()
        self.rec_bg = SpeechRecognizer()
        
        # Inizializzazione RAG con logging
        if documents:
            logging.info("Initializing RAG with documents...")
            self.doc_retriever = RagDocumentRetriever(documents)
        else:
            self.doc_retriever = None
            logging.warning("No documents provided for RAG")

        # Eventi e lock
        self.interrupt_event = Event()
        self.shutdown_event = Event()
        self.sound_queue = queue.Queue()
        self.audio_lock = Lock()
        self.speaking = Event()
        
        # Storia della conversazione
        self.conversation_history = []
        
        # Controllo duplicati
        self.last_command = ""
        self.last_command_time = 0
        self.command_cooldown = 1.5

        # Avvio thread per suoni
        self.sound_thread = Thread(target=self._sound_worker, daemon=True)
        self.sound_thread.start()

    def _sound_worker(self):
        while not self.shutdown_event.is_set():
            try:
                snd = self.sound_queue.get(timeout=0.5)
                if snd:
                    self.sound.play_sound(snd)
                    self.sound_queue.task_done()
            except queue.Empty:
                continue
        logging.info("Sound worker terminated")

    def play_sound(self, sound_id):
        if not self.shutdown_event.is_set():
            self.sound_queue.put_nowait(sound_id)

    def listen_wake(self):
        max_retries = 3
        for attempt in range(max_retries):
            if self.shutdown_event.is_set():
                return False
            try:
                wake_word_detected = self.rec_main.listen_for_wake_word(config.WAKE_WORD)
                if wake_word_detected:
                    logging.info(f"Wake word detected on attempt {attempt + 1}")
                    return True
            except Exception as e:
                logging.error(f"Wake-listen error (attempt {attempt + 1}): {e}")
                time.sleep(0.1)
        return False

    def is_duplicate_command(self, command):
        current_time = time.time()
        if (command == self.last_command and 
            current_time - self.last_command_time < self.command_cooldown):
            return True
        self.last_command = command
        self.last_command_time = current_time
        return False

    def listen_command(self, timeout=None):
        try:
            with self.audio_lock:
                command = self.rec_main.listen(timeout=timeout)
                if command and self.is_duplicate_command(command):
                    logging.debug(f"Duplicate command ignored: {command}")
                    return ""
                return command
        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            logging.error(f"Listen error: {e}")
            return ""

    def process(self, command: str):
        if not command:
            return
            
        self.interrupt_event.clear()
        self.speaking.set()
        
        try:
            # Aggiunge il comando alla storia
            self.conversation_history.append({"role": "user", "content": command})
            
            # Recupera documenti dal RAG
            context = ""
            rag_docs = []
            
            if self.doc_retriever:
                try:
                    rag_docs = self.doc_retriever.retrieve(command)
                    if rag_docs:
                        logging.info(f"RAG found {len(rag_docs)} relevant documents")
                        # Prendi solo il testo (secondo elemento della tupla)
                        docs_text = "\n".join(testo for _, testo, _ in rag_docs)
                        self.api.talk_direct_tts(
                            docs_text,
                            tts_callback=lambda text: self.tts.speak(text)
                        )
                        self.conversation_history.append({"role": "assistant", "content": docs_text})
                except Exception as e:
                    logging.error(f"RAG retrieval error: {e}")
            else:
                # Aggiunge la storia della conversazione
                #conversation_context = "\n".join([
                #    f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}"
                #    for item in self.conversation_history[-10:]
                #])
                #context += f"\nConversation history:\n{conversation_context}"
                
                # Processa il comando
                if "think" in command.lower():
                    logging.info(f"Processing 'think' command with {len(rag_docs)} RAG docs")
                    resp = self.api.think(command, context)
                    self._speak_interruptible(resp)
                    self.conversation_history.append({"role": "assistant", "content": resp})
                else:
                    logging.info(f"Processing streaming command with {len(rag_docs)} RAG docs")
                    response_buffer = []
                    
                    def capture_response(chunk):
                        if self.interrupt_event.is_set():
                            return False
                        # Log della risposta per debug
                        logging.debug(f"TTS chunk: {chunk}")
                        response_buffer.append(chunk)
                        self.tts.speak(chunk)
                        return True
                        
                    self.api.talk_stream_tts_phrase(
                        command,
                        context,
                        tts_callback=capture_response,
                        command_callback=lambda c: self.interrupt_event.set()
                    )
                    
                    full_response = " ".join(response_buffer)
                    if full_response:
                        self.conversation_history.append(
                            {"role": "assistant", "content": full_response}
                        )
        finally:
            self.speaking.clear()

    def _speak_interruptible(self, text):
        for sent in text.split(". "):
            if self.interrupt_event.is_set():
                break
            sent = sent.strip()
            if sent:
                logging.debug(f"Speaking: {sent}")
                self.tts.speak(sent)

    def run(self):
        logging.info("Assistant started, waiting for wake word")
        
        while not self.shutdown_event.is_set():
            if self.listen_wake():
                self.play_sound(config.WAKE_SOUND)
                time.sleep(0.5)  # Cooldown dopo wake word
                
                while not self.shutdown_event.is_set():
                    logging.info("Listening for command...")
                    cmd = self.listen_command(config.LISTEN_TIMEOUT)
                    
                    if not cmd:
                        logging.info("Timeout or empty command")
                        self.play_sound(config.TIMEOUT_SOUND)
                        break
                        
                    if any(k in cmd.lower() for k in ("stop", "exit", "quit", "termina")):
                        logging.info(f"Exit command received: {cmd}")
                        self.play_sound(config.STOP_SOUND)
                        break
                        
                    logging.info(f"Processing command: '{cmd}'")
                    self.process(cmd)
                    time.sleep(1.0)  # Cooldown dopo elaborazione
                    
            time.sleep(0.1)
            
        self.shutdown()

    def shutdown(self):
        logging.info("Shutting down Assistant")
        self.shutdown_event.set()
        self.interrupt_event.set()
        if self.sound_thread.is_alive():
            self.sound_thread.join(timeout=1.0)