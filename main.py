import openai
import pyaudio
import numpy as np
import wave
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import asyncio
import concurrent.futures
import time
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import webbrowser
import numpy as np
import pvporcupine, struct, os
from dotenv import load_dotenv

load_dotenv()

PORCUPINE_ACCESS_KEY = os.getenv('PORCUPINE_ACCESS_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

client = openai.OpenAI(api_key=openai_api_key)
MODEL = "gpt-4o"

elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)

audio = pyaudio.PyAudio()

def setup_wake_word_detection(wake_word="computer"):
    try:
        porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keywords=[wake_word]
        )
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
            input_device_index=None)  # You might need to specify the correct input device index
        return porcupine, pa, audio_stream
    except Exception as e:
        print(f"An error occurred while setting up Porcupine: {e}")
        return None, None, None

def detect_wake_word(porcupine, audio_stream):
    print(f"Listening for wake word...")
    while True:
        try:
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
                return True
            time.sleep(0.01)  # Add a small delay to prevent excessive CPU usage
        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                print("Input overflow, ignoring this chunk of audio.")
            else:
                print(f"An error occurred while processing audio: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(1)  # Wait a bit before retrying

def setup_spotify():
    auth_manager = SpotifyOAuth(
        client_id=spotify_client_id,
        client_secret=spotify_client_secret,
        redirect_uri="http://localhost:8888/callback",
        scope="user-library-read user-modify-playback-state user-read-playback-state",
        open_browser=False
    )
    
    # Check if there's a cached token
    token_info = auth_manager.get_cached_token()
    
    if not token_info:
        # If there's no cached token, get the auth URL
        auth_url = auth_manager.get_authorize_url()
        print(f"Please go to this URL to authorize the application: {auth_url}")
        webbrowser.open(auth_url)
        
        # Wait for the user to authorize and get the response URL
        response = input("Enter the URL you were redirected to: ")
        
        # Extract the code from the response URL
        code = auth_manager.parse_response_code(response)
        
        # Get the access token
        token_info = auth_manager.get_access_token(code)
    
    return spotipy.Spotify(auth_manager=auth_manager)

sp = setup_spotify()

class CalendarEvent:
    def __init__(self, event_name, event_date_and_time, event_place='', event_note='', event_priority=''):
        self.event_name = event_name
        self.event_date_and_time = event_date_and_time
        self.event_place = event_place
        self.event_note = event_note
        self.event_priority = event_priority

    def __repr__(self):
        return (f"CalendarEvent(event_name={self.event_name}, event_date_and_time={self.event_date_and_time}, "
                f"event_place={self.event_place}, event_note={self.event_note}, event_priority={self.event_priority})")

    def get_details(self):
        return {
            "event_name": self.event_name,
            "event_date_and_time": self.event_date_and_time,
            "event_place": self.event_place,
            "event_note": self.event_note,
            "event_priority": self.event_priority
        }
    
    @classmethod
    def from_dict(cls, event_dict):
        return cls(**event_dict)

class Calendar:
    def __init__(self):
        self.events = []
        self.load_events()

    def add_event(self, event):
        self.events.append(event)
        self.events.sort(key=lambda x: datetime.fromisoformat(x.event_date_and_time))
        self.save_events()

    def delete_event(self, event_name):
        self.events = [event for event in self.events if event.event_name != event_name]
        self.save_events()

    def get_next_event(self):
        now = datetime.now()
        future_events = [event for event in self.events if datetime.fromisoformat(event.event_date_and_time) > now]
        return future_events[0] if future_events else None

    def get_event_details(self, event_name):
        for event in self.events:
            if event.event_name == event_name:
                return event.get_details()
        return None
    
    def save_events(self):
        with open('calendar_events.txt', 'w') as f:
            json.dump([event.get_details() for event in self.events], f)

    def load_events(self):
        if os.path.exists('calendar_events.txt'):
            with open('calendar_events.txt', 'r') as f:
                events_data = json.load(f)
                self.events = [CalendarEvent.from_dict(event_dict) for event_dict in events_data]
            self.events.sort(key=lambda x: datetime.fromisoformat(x.event_date_and_time))

    def get_all_event_names(self):
        return [event.event_name for event in self.events]

    def __repr__(self):
        return f"Calendar(events={self.events})"

class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.current_date = datetime.now()
        self.calendar = Calendar()
        self.contacts = []
        self.tasks = []

    def __repr__(self):
        return (f"User(username={self.username}, email={self.email}, current_date={self.current_date}, "
                f"calendar={self.calendar}, contacts={self.contacts}, tasks={self.tasks})")

    def add_event(self, event_name, event_date_and_time, event_place='', event_note='', event_priority=''):
        event = CalendarEvent(event_name, event_date_and_time, event_place=event_place, event_note=event_note, event_priority=event_priority)
        self.calendar.add_event(event)
        return f"Event {event_name} added to the calendar."

    def delete_event(self, event_name):
        self.calendar.delete_event(event_name)
        return f"Event {event_name} deleted from the calendar."

    def get_next_event(self):
        next_event = self.calendar.get_next_event()
        return next_event.get_details() if next_event else "No upcoming events."

    def get_event_details(self, event_name):
        event_details = self.calendar.get_event_details(event_name)
        return event_details if event_details else f"No details found for event {event_name}."
    
    def get_full_calendar(self):
        return self.calendar.get_all_event_names()

    def dummy_function(self):
        return "No specific function called."
    
    def spotify_search_and_play(self, query, device_id=None):
        # First, check for available devices
        devices = sp.devices()
        if not devices['devices']:
            return "No Spotify devices found. Please open Spotify on a device and try again."

        # If no specific device_id is provided, use the first available device
        if not device_id:
            device_id = devices['devices'][0]['id']

        # Ensure the selected device is active
        try:
            sp.transfer_playback(device_id, force_play=True)
        except spotipy.exceptions.SpotifyException as e:
            return f"Error activating device: {str(e)}"

        # Wait for the device to become active
        time.sleep(2)

        # Now search and play
        results = sp.search(q=query, type='track', limit=1)
        if not results['tracks']['items']:
            return f"Sorry, I couldn't find any tracks matching '{query}'"

        track = results['tracks']['items'][0]
        try:
            sp.start_playback(device_id=device_id, uris=[track['uri']])
            
            # Wait for a moment to let Spotify process the command
            time.sleep(2)
            
            # Check if the track is actually playing
            playback_state = sp.current_playback()
            if not playback_state or not playback_state['is_playing']:
                # If not playing, try to resume playback
                sp.start_playback(device_id=device_id)
                time.sleep(1)
                
                # Check again
                playback_state = sp.current_playback()
                if not playback_state or not playback_state['is_playing']:
                    return f"Started {track['name']} by {track['artists'][0]['name']}, but playback may not have begun. Please check your Spotify app."
            
            return f"Now playing: {track['name']} by {track['artists'][0]['name']} on the selected device."
        except spotipy.exceptions.SpotifyException as e:
            return f"Error playing track: {str(e)}"

    def spotify_pause(self, device_id=None):
        try:
            sp.pause_playback(device_id=device_id)
            return "Playback paused on the selected device."
        except spotipy.exceptions.SpotifyException as e:
            return f"Error pausing playback: {str(e)}"

    def spotify_play(self, uri=None, device_id=None):
        try:
            if uri:
                sp.start_playback(device_id=device_id, uris=[uri])
            else:
                sp.start_playback(device_id=device_id)
            return "Playback started on the selected device."
        except spotipy.exceptions.SpotifyException as e:
            return f"Error starting playback: {str(e)}"

    def execute_function(self, function_name, args):
        if hasattr(self, function_name):
            func = getattr(self, function_name)
            if callable(func):
                return func(**args)
        return "Function not found or not callable."
    
    def end_session(self):
        return "Ending session"

def record_audio(filename, silence_threshold=500, silence_duration=0.8):
    chunk = 256  # Record in smaller chunks to reduce latency
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second

    stream = audio.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

    print("Recording...")
    frames = []
    silent_chunks = 0
    silence_limit = int(silence_duration * fs / chunk)

    while True:
        data = stream.read(chunk)
        frames.append(data)

        audio_data = np.frombuffer(data, dtype=np.int16)
        silence = np.mean(np.abs(audio_data)) < silence_threshold

        if silence:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks > silence_limit:
            silent_chunks = 0
            grace_period = 3
            grace_start = time.time()
            while time.time() - grace_start < grace_period:
                data = stream.read(chunk)
                frames.append(data)

                audio_data = np.frombuffer(data, dtype=np.int16)
                silence = np.mean(np.abs(audio_data)) < silence_threshold

                if not silence:
                    silent_chunks = 0
                    break

                silent_chunks += 1
                if silent_chunks > silence_limit:
                    break

            if silent_chunks > silence_limit:
                print("Silence detected, stopping recording.")
                break

    print("Finished recording.")

    stream.stop_stream()
    stream.close()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

async def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        transcript = await loop.run_in_executor(
            executor, 
            lambda: client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        )
    return transcript.text

async def get_output_response(conversation):
    completion = await loop.run_in_executor(
        executor, 
        lambda: client.chat.completions.create(
            model=MODEL,
            messages=conversation
        )
    )
    return completion.choices[0].message.content

import json
import json

import json

async def decide_tool_call(user, recent_user_message):
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    event_names = user.calendar.get_all_event_names()

    spotify_devices = sp.devices()['devices']
    devices_info = [{'id': d['id'], 'name': d['name'], 'is_active': d['is_active']} for d in spotify_devices]

    system_prompt = f"""
    You are a function calling assistant. Based on the user's request, decide which functions to call and with what arguments. Return the decisions as a single JSON object where 
    each key is a unique identifier and each value is another object containing the function name and its arguments. If you think the user didn't intend for anything to be called,
    call the dummy function. If you think the user is done talking to you or wants the session to be over, call end_session(). If you suspect even a little bit that the user wants
    event details, call the get_event_details function pretty liberally, for all the events the user asks about. When moving user events around the calendar, make sure you prioritize
    keeping all the event info correct.
    
    Example of expected output format:
    {{
        "call1": {{
            "function": "add_event",
            "args": {{
                "event_name": "Meeting with Tom",
                "event_date_and_time": "2024-06-25 08:00:00",
                "event_place": "Office",
                "event_priority": "high"
            }}
        }},
        "call2": {{
            "function": "add_event",
            "args": {{
                "event_name": "Jogging",
                "event_date_and_time": "2024-06-25 18:00:00",
                "note": "beat best mile time of 8 minutes"
            }}
        }}
    }}

    Example #2 of expected output format:
    {{
        "call1": {{
            "function": "spotify_search_and_play",
            "args": {{
                "query": "Bohemian Rhapsody by Queen"
            }}
        }}
    }}

    Available functions include (=SOMETHING means optional, every other argument without =SOMETHING is required):
    - add_event(event_name, event_date_and_time, event_place='', event_note='', event_priority='')
    - delete_event(event_name: str)
    - get_next_event()
    - get_event_details(event_name: str)
    - get_full_calendar()
    - spotify_search_and_play(query: str, device_id: str = None)
    - spotify_pause(device_id: str = None)
    - spotify_play(uri: str = None, device_id: str = None)
    - dummy_function()
    - end_session()
    
    Spotify Function Usage:
    - If the user specifies a device, include the device_id in the function call.
    - If no device is specified, you can omit the device_id parameter, and the function will use the default active device.
    - spotify_search_and_play is used for playing specific songs or artists.
    - spotify_play is used to resume playback or play a specific URI.
    - spotify_pause is used to pause playback.

    Available Spotify Devices: 
    {devices_info}

    When selecting a device, use the 'id' field from the devices list above. Choose the device that best matches the user's request.
    If you can't find a good device match, just pick the first device.

    The current date and time is {current_datetime}.
    Current events in the user's calendar: {event_names}
    """
    
    completion = await loop.run_in_executor(
        executor,
        lambda: client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": recent_user_message + " Decide what functions to call based on my request and return them in the specified JSON object format."}
            ],
            temperature=1,
            max_tokens=512,
            stop=None,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"}
        )
    )
    return json.loads(completion.choices[0].message.content)

async def generate_speech(text):
    audio_stream = await loop.run_in_executor(
        executor, 
        lambda: elevenlabs_client.generate(
            text=text,
            voice="Rachel",
            model="eleven_multilingual_v2",
            stream=True
        )
    )
    return audio_stream

async def handle_interaction(user):
    record_audio("user_input.wav", silence_threshold=500, silence_duration=2)
    user_message = await transcribe_audio("user_input.wav")
    print(f"User: {user_message}")
    conversation.append({"role": "user", "content": user_message})

    tool_calls = await decide_tool_call(user, user_message + " Return this in JSON object format.")
    results = []
    for call_id, tool_call in tool_calls.items():
        function_name = tool_call['function']
        function_args = tool_call.get('args', {})
        try:
            result = user.execute_function(function_name, function_args)
            results.append(str(result))  # Convert result to string here
        except Exception as e:
            result = f"Error executing tool call: {str(e)}"
            results.append(result)
    
    tool_result = " | ".join(results)  # Now all elements are strings
    print(f"Tool Results: {tool_result}")

    assistant_message = await get_output_response(conversation + [{"role": "system", "content": f"Tool result (use this only if necessary to help assist the user): {tool_result}"}])
    print(f"Assistant: {assistant_message}")
    conversation.append({"role": "assistant", "content": assistant_message})

    audio_stream = await generate_speech(assistant_message)
    stream(audio_stream)
    if "Ending session" in tool_result:
        return False
    return True
    

conversation = [
    {"role": "system", "content": "You are a friendly and helpful assistant. Respond to the user's queries concisely and directly, but feel free to add a touch of personality occasionally. Keep your responses short and to the point most of the time. Your goal is to make the conversation feel natural and human-like. If the result of a function call are provided, use that as part of your response. I.E. if you get a function result saying that the users requested song is playing on a specific device, say that you are playing it for them."}
]

executor = concurrent.futures.ThreadPoolExecutor()
loop = asyncio.get_event_loop()

sp = setup_spotify()

try:
    porcupine, pa, audio_stream = setup_wake_word_detection()
    if porcupine is None:
        print("Failed to initialize wake word detection. Exiting.")
        raise SystemExit

    user = User(username="Athul", email="athul@example.com")

    while True:
        if detect_wake_word(porcupine, audio_stream):
            print("Starting conversation...")
            session_active = True 
            while session_active:
                session_active = loop.run_until_complete(handle_interaction(user))
            print("Session ended. Listening for wake word again...")

except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    if 'porcupine' in locals() and porcupine is not None:
        porcupine.delete()
    if 'audio_stream' in locals() and audio_stream is not None:
        audio_stream.stop_stream()
        audio_stream.close()
    if 'pa' in locals() and pa is not None:
        pa.terminate()
