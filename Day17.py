# Voice Assistant with Robust Command Matching
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
import string

# Initialize TTS engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Change index if needed
engine.setProperty('rate', 180)  # Slower rate for clarity

def speak(text):
    """Speak the given text aloud."""
    engine.say(text)
    engine.runAndWait()

def take_command():
    """Listen to microphone input and return recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        while True:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=8)
                print("Recognizing...")
                command = recognizer.recognize_google(audio)
                print(f"User said: {command}")
                return command.lower()
            except sr.WaitTimeoutError:
                print("No speech detected, trying again...")
            except sr.UnknownValueError:
                print("Sorry, I did not understand that. Please repeat.")
            except sr.RequestError:
                print("Network error.")
                return None
            except Exception as e:
                print("Microphone error:", e)
                return None

def respond(command):
    """Respond to recognized commands with robust matching."""
    if command is None:
        return

    # Clean and split command
    command_clean = command.translate(str.maketrans('', '', string.punctuation)).lower()
    words = command_clean.split()

    # Greetings
    if any(word in ["hi", "hello", "hey"] for word in words):
        speak("Hello! How can I assist you today?")
        return

    # Time
    if any("time" in word for word in words):
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")
        return

    # Search
    if "search" in command_clean:
        speak("What would you like to search for?")
        search_query = take_command()
        if search_query:
            speak(f"Searching for {search_query}")
            webbrowser.open(f"https://www.google.com/search?q={search_query}")
        return

    # Open applications
    if "notepad" in command_clean:
        speak("Opening Notepad")
        os.system("notepad")
        return
    if "calculator" in command_clean:
        speak("Opening Calculator")
        os.system("calc")
        return

    # Exit
    if any(word in command_clean for word in ["bye", "exit", "quit"]):
        speak("Goodbye! Have a great day.")
        exit()

    # Unknown command fallback
    speak("I'm sorry, I don't know that command. Try saying 'time', 'search', or 'open notepad'.")

def run_assistant():
    """Main loop to run the assistant."""
    speak("Hello, I am your assistant. How can I help you?")
    while True:
        command = take_command()
        respond(command)

if __name__ == "__main__":
    run_assistant()
