# pip install google-generativeai pyaudio speechrecognition

import asyncio
import traceback
import os
import pyaudio
import wave
import speech_recognition as sr
from google import genai
from google.genai import types
import io

# -------------------------------------------------------------------
# Audio parameters
# -------------------------------------------------------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

# -------------------------------------------------------------------
# API key and client setup
# -------------------------------------------------------------------
os.environ["GOOGLE_GEMINI_API_KEY"] = "GOOGLE_GEMINI_API_KEY"
client = genai.Client(
    api_key=os.environ["GOOGLE_GEMINI_API_KEY"],
    http_options={"api_version": "v1beta"},
)

# -------------------------------------------------------------------
# LiveConnectConfig: audio-only responses
# -------------------------------------------------------------------
CONFIG = types.LiveConnectConfig(
    # only configure Gemini's output as audio
    response_modalities=[types.Modality.AUDIO],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

pya = pyaudio.PyAudio()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# -------------------------------------------------------------------
# XArm Control Import and Setup
# -------------------------------------------------------------------
import sys
import math
import time
import queue
import datetime
import random
import threading
from xarm import version
from xarm.wrapper import XArmAPI

class RobotMain(object):
    """Robot Main Class"""
    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._ignore_exit_state = False
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()
    
    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)
    
    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
    
    # Register state changed callback
    def _state_changed_callback(self, data):
        if not self._ignore_exit_state and data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)
    
    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))
    
    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive
    
    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)
    
    @property
    def arm(self):
        return self._arm
    
    @property
    def VARS(self):
        return self._vars
    
    @property
    def FUNCS(self):
        return self._funcs
    
    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._ignore_exit_state:
                return True
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False
    
    # Robot Main Run
    def run(self):
        try:
            code = self._arm.set_gripper_position(850, wait=True, speed=5000, auto_enable=True)
            if not self._check_code(code, 'set_gripper_position'):
                return
            code = self._arm.set_servo_angle(angle=[-127.2, -5.6, -146.5, -2.0, 60.2, -0.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-128.5, 11.5, -144.9, -2.0, 46.4, -0.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_gripper_position(450, wait=True, speed=5000, auto_enable=True)
            if not self._check_code(code, 'set_gripper_position'):
                return
            code = self._arm.set_servo_angle(angle=[-128.3, -11.2, -145.1, -2.0, 65.0, -0.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-26.4, 90.7, -139.2, -1.9, -39.7, -0.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)

class AudioBuffer:
    """A helper class for buffering and processing audio chunks"""
    def __init__(self, chunk_size, sample_rate, buffer_duration=2.0):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer = bytearray()
        self.max_buffer_size = int(sample_rate * buffer_duration) * 2  # 2 bytes per sample for FORMAT=paInt16
    
    def add_chunk(self, chunk):
        """Add a chunk to the buffer and return True if buffer is ready for processing"""
        self.buffer.extend(chunk)
        return len(self.buffer) >= self.max_buffer_size
    
    def get_wav_data(self):
        """Convert buffer to WAV format and return it as bytes"""
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for paInt16
                wf.setframerate(self.sample_rate)
                wf.writeframes(self.buffer)
            
            # Reset buffer with some overlap
            overlap = self.buffer[-self.chunk_size*2:]
            self.buffer = bytearray(overlap)
            
            # Get WAV data
            wav_buffer.seek(0)
            return wav_buffer.read()

class AudioLoop:
    def __init__(self):
        self.session = None
        self.audio_in_queue = None
        self.out_queue = None
        self.audio_stream = None
        self.audio_buffer = AudioBuffer(CHUNK_SIZE, SEND_SAMPLE_RATE)
        self.trigger_words = ["drink", "pass me", "give me", "pepsi", "get me"]
        self.is_xarm_running = False
        self.xarm_lock = asyncio.Lock()
    
    async def listen_audio(self):
        """Listen to microphone and process audio chunks"""
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        # avoid overflow exceptions
        kwargs = {"exception_on_overflow": False}
        
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            
            # Send to Gemini
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            
            # Process for speech recognition
            if self.audio_buffer.add_chunk(data):
                wav_data = self.audio_buffer.get_wav_data()
                asyncio.create_task(self.process_speech(wav_data))
    
    async def process_speech(self, wav_data):
        """Process speech in WAV format and check for triggers"""
        try:
            # Use Speech Recognition to transcribe audio
            with io.BytesIO(wav_data) as wav_io:
                with sr.AudioFile(wav_io) as source:
                    audio_data = recognizer.record(source)
                    # Use Google's speech recognition
                    text = recognizer.recognize_google(audio_data)
                    print(f"Recognized: {text}")
                    
                    # Check for trigger words
                    text = text.lower()
                    if any(trigger in text for trigger in self.trigger_words):
                        print(f"Trigger detected in: {text}")
                        
                        # Only run xarm if it's not already running
                        async with self.xarm_lock:
                            if not self.is_xarm_running:
                                self.is_xarm_running = True
                                # Run the xarm in a separate thread to avoid blocking
                                await asyncio.to_thread(self.run_xarm)
                                self.is_xarm_running = False
        except sr.UnknownValueError:
            # Speech was unintelligible
            pass
        except Exception as e:
            print(f"Error in speech recognition: {e}")

    async def send_realtime(self):
        """Send audio to Gemini"""
        while True:
            packet = await self.out_queue.get()
            await self.session.send(input=packet)

    async def receive_audio(self):
        """Receive audio responses from Gemini"""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.data:
                    # incoming PCM bytes
                    self.audio_in_queue.put_nowait(response.data)

    async def play_audio(self):
        """Play audio responses from Gemini"""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    def run_xarm(self):
        """Run the xArm to fetch a drink"""
        print("Starting xArm to fetch drink...")
        try:
            # Initialize and run the xArm
            arm = XArmAPI('192.168.1.220', baud_checkset=False)
            robot_main = RobotMain(arm)
            robot_main.run()
            print("xArm completed the sequence.")
        except Exception as e:
            print(f"Error running xArm: {e}")
            traceback.print_exc()

    async def run(self):
        """Main method to run the audio loop"""
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # start tasks
                tg.create_task(self.listen_audio())
                tg.create_task(self.send_realtime())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # run indefinitely until you Ctrl+C
                await asyncio.Future()

        except asyncio.CancelledError:
            pass
        except Exception as eg:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(type(eg), eg, eg.__traceback__)

if __name__ == "__main__":
    loop = AudioLoop()
    asyncio.run(loop.run())