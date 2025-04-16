import logging
import time
from multiprocessing import Process, Event, Queue
from threading import Thread, Lock
from audio.tts import Pyttsx3TTS
from audio.sound_player import SoundPlayer
from api.api_client import APIClient
from recognizer.speech_recognizer import SpeechRecognizer
from document.document_retriever import DocumentRetriever
import speech_recognition as sr
import config

class VoiceAssistant:
    def __init__(self, documents: dict = None):
        self.documents = documents
        self.tts = Pyttsx3TTS()
        self.sound_player = SoundPlayer()
        self.api_client = APIClient()
        
        # Separate recognizers for main and background listening
        self.main_recognizer = SpeechRecognizer()
        self.background_recognizer = SpeechRecognizer()
        
        self.document_retriever = DocumentRetriever(documents) if documents else None
        
        # Control flags for interruption handling
        self.interrupt_event = Event()
        self.command_queue = Queue()
        self.speaking = Event()
        self.listening_for_wake_word = Event()
        
        # Mutex for audio resource access
        self.audio_lock = Lock()
        
    def process_command(self, command: str):
        if not command:
            logging.warning("No command to process.")
            return
        
        context = ""
        if self.document_retriever:
            retrieved_docs = self.document_retriever.retrieve(command)
            context = "\n".join([doc for _, doc in retrieved_docs])
        
        # Set the speaking flag to indicate TTS is active
        self.speaking.set()
        
        try:
            # If command contains 'think' use the think model, otherwise use streaming TTS
            if "think" in command.lower() or "pensa" in command.lower():
                response = self.api_client.think(command, context)
                # Use the interruptible speak method
                self.speak_with_interruption(response)
            else:
                # Modified stream TTS to support interruption
                self.stream_tts_with_interruption(command, context)
        finally:
            # Clear the speaking flag when done
            self.speaking.clear()
    
    def speak_with_interruption(self, text):
        """
        Speaks the text while allowing for interruption.
        """
        # Reset the interrupt flag
        self.interrupt_event.clear()
        
        # Split the response into sentences or smaller chunks
        chunks = self.split_into_chunks(text)
        
        for chunk in chunks:
            # Check if interrupted before speaking each chunk
            if self.interrupt_event.is_set():
                logging.info("Speech interrupted.")
                break
            
            # Speak the current chunk
            self.tts.speak(chunk)
    
    def split_into_chunks(self, text, max_length=150):
        """
        Split text into smaller, sentence-aware chunks for better interruption handling.
        """
        # Simple implementation - can be improved with sentence detection
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def stream_tts_with_interruption(self, command, context):
        """
        Streams TTS response with interruption support.
        """
        # Custom callback that checks for interruption
        def interruptible_tts_callback(phrase):
            if not self.interrupt_event.is_set():
                self.tts.speak(phrase)
            return not self.interrupt_event.is_set()  # Return False to stop streaming if interrupted
        
        # Call the API client with our custom callback
        self.api_client.talk_stream_tts_phrase(command, context, tts_callback=interruptible_tts_callback)
        #self.api_client.talk(command, context)
    
    def background_listener(self):
        """
        Continuous background listening process that can detect commands
        even while TTS is speaking.
        """
        while True:
            if self.speaking.is_set() and not self.listening_for_wake_word.is_set():
                try:
                    # Only attempt to listen if not in wake word detection mode
                    command = self.background_recognizer.listen(timeout=3)  # Short timeout for responsiveness
                    
                    # If we got a valid command and assistant is speaking
                    if command:
                        logging.info(f"Background listener detected command: {command}")
                        
                        # Check if it's an interruption command
                        if any(keyword in command.lower() for keyword in ["stop", "wait", "interrupt", "fermati", "basta"]):
                            # Signal interruption
                            self.interrupt_event.set()
                            self.play_sound_async(config.INTERRUPT_SOUND)
                        
                        # Add the command to the queue for processing after current speech ends
                        self.command_queue.put(command)
                
                except sr.WaitTimeoutError:
                    # Just continue listening on timeout
                    pass
                except Exception as e:
                    logging.error(f"Error in background listener: {e}")
                    # Add a small delay to prevent rapid error loops
                    time.sleep(0.5)
            else:
                # If not speaking or we're in wake word mode, just sleep briefly
                time.sleep(0.1)
    
    def play_sound_async(self, sound_file: str):
        """
        Plays a sound in a separate process so that it doesn't block the main thread.
        """
        process = Process(target=self.sound_player.play_sound, args=(sound_file,))
        process.start()
    
    def listen_with_recovery(self, timeout=None):
        """
        Listen for commands with error recovery.
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.audio_lock:
                    return self.main_recognizer.listen(timeout=timeout)
            except sr.WaitTimeoutError:
                # This is an expected exception when timeout occurs
                return ""
            except Exception as e:
                logging.error(f"Error during listening: {e}")
                retry_count += 1
                time.sleep(0.5)  # Brief pause before retry
                
                # Recreate the recognizer if we're having persistent issues
                if retry_count == max_retries - 1:
                    logging.info("Recreating speech recognizer due to persistent errors")
                    self.main_recognizer = SpeechRecognizer()
        
        return ""  # Return empty string if all retries fail
    
    def listen_for_wake_word_safely(self, wake_word):
        """
        Listen for wake word with error recovery.
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.listening_for_wake_word.set()
                with self.audio_lock:
                    result = self.main_recognizer.listen_for_wake_word(wake_word)
                self.listening_for_wake_word.clear()
                return result
            except Exception as e:
                logging.error(f"Error listening for wake word: {e}")
                retry_count += 1
                time.sleep(0.5)  # Brief pause before retry
                
                # Recreate the recognizer if we're having persistent issues
                if retry_count == max_retries - 1:
                    logging.info("Recreating speech recognizer due to persistent errors")
                    self.main_recognizer = SpeechRecognizer()
                
                self.listening_for_wake_word.clear()
        
        return False  # Return False if all retries fail
    
    def run(self):
        # Start the background listener thread
        listener_thread = Thread(target=self.background_listener, daemon=True)
        listener_thread.start()
        
        # Speak a welcome message before starting to listen for the wake word
        welcome_message = f"Hi there! I'm {config.WAKE_WORD}"
        self.tts.speak(welcome_message)
        logging.info("Welcome message delivered. Waiting for wake word.")
        
        while True:
            try:
                # Wait for the wake word
                if self.listen_for_wake_word_safely(config.WAKE_WORD):
                    self.play_sound_async(config.WAKE_SOUND)
                    logging.info("Entering command mode.")
                    
                    # Enter command mode: continuously listen for commands
                    command_mode_active = True
                    while command_mode_active:
                        try:
                            # Check if we have a queued command from the background listener
                            if not self.command_queue.empty():
                                command = self.command_queue.get()
                                logging.info(f"Processing queued command: {command}")
                            else:
                                command = self.listen_with_recovery(timeout=config.LISTEN_TIMEOUT)
                            
                            # If no command was recognized or stop command detected
                            if not command:
                                logging.info("No command received within timeout.")
                                self.play_sound_async(config.TIMEOUT_SOUND)
                                command_mode_active = False
                            elif any(word in command.lower() for word in ["stop", "exit", "quit", "termina"]):
                                self.play_sound_async(config.STOP_SOUND)
                                logging.info("'Stop' command detected, exiting command mode.")
                                command_mode_active = False
                            else:
                                # Process the received command using the APIClient
                                self.process_command(command)
                            
                        except sr.WaitTimeoutError:
                            logging.info("No command received within timeout.")
                            self.play_sound_async(config.TIMEOUT_SOUND)
                            # Exit command mode after a timeout
                            command_mode_active = False
                        
                        # Add a small delay between command processing
                        time.sleep(0.1)
                
                # Brief pause before listening for wake word again
                time.sleep(0.5)
            
            except Exception as e:
                logging.error(f"Error in the main loop: {e}")
                # Add recovery mechanism
                time.sleep(1)  # Pause before retrying