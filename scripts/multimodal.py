# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""

import asyncio
import base64
import io
import os
import sys
import traceback

import cv2
import pyaudio
import PIL.Image
import mss
import numpy as np

import argparse
import pyrealsense2 as rs
from google.genai import types

from google import genai
import tkinter as tk
from threading import Thread

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    from exceptiongroup import ExceptionGroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

DEFAULT_MODE = "camera"

client = genai.Client(api_key='AIzaSyCNu5AJ9QXMYX2uNxVCM4kmNTH-V9uvA50', http_options={"api_version": "v1alpha"})

# While Gemini 2.0 Flash is in experimental preview mode, only one of AUDIO or
# TEXT may be passed here.
CONFIG = {
    "system_instruction": types.Content(
        parts=[
            types.Part(
                text="""
                You are a six degree of freedom robotic arm, and you are looking through a camera on the end effector. You are mounted on a table, where there is a plate and a cup. 
                There is external code which is programmed to feed the contents of the plate, one item at a time, to a user with a spinal cord injury. It will also bring the cup to the user when asked.
                You have no autonomy over the actions of the robot; your purpose is to explain what is happening throughout the process, and provide light-hearted conversation to the user. 
                You should also be able to answer questions that the user asks you. I will now provide background on how the system works.

                Background: The user has picked up to three food items from a menu, and the selected food items are on the plate. Do not list the menu items while introducing yourself.
                This is a list of the menu items:
                ['pretzel bites','celery', 'carrots','pretzel rods','sushi','green grapes','chicken nuggets', 'chocolate','donuts', ' yellow popcorn', 'gummies']

                There will never be a food item on the plate that is not on this menu. Under no circumstances refer to food that is not on this menu. When the session begins the robot will begin the feeding sequence, 
                which can take some time. This is how the sequence works: a picture of the table is taken, and ChatGPT is used to identify the food items on the plate. After these items are identified,
                calculations are performed to find the distance of the food relative to the end effector. Then, path planning is performed and the robot brings the food item to the user's mouth.
                At any point, the user can ask for a drink. If the drink is asked for while food is in the gripper, notify the user that the robot will grab the drink after you finish feeding the food item.

                If the user asks you to pick up a certain food item, tell them that it is programmed to bring food randomly to the mouth. 
                Sometimes it will try to pick up a food item between the grippers and miss, or drop it. This is okay, and you should tell the user it missed an item and is going to try again.
                """
            )
        ]
    ),
    "response_modalities": ["AUDIO"],
}

pya = pyaudio.PyAudio()

class RAF_GUI:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=200, height=200)
        self.canvas.pack()
        self.oval = self.canvas.create_oval(50, 50, 150, 150, fill="blue")
        self.pulsing = False
        self.pulse()

    def pulse(self):
        if self.pulsing:
            self.canvas.scale(self.oval, 100, 100, 1.1, 1.1)
            self.canvas.after(100, self.shrink)
        else:
            self.canvas.after(100, self.pulse)

    def shrink(self):
        if self.pulsing:
            self.canvas.scale(self.oval, 100, 100, 0.9, 0.9)
            self.canvas.after(100, self.pulse)
        else:
            self.canvas.after(100, self.shrink)

    def start_pulsing(self):
        self.pulsing = True

    def stop_pulsing(self):
        self.pulsing = False

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        # Check for available devices
        if not self.list_devices():
            raise RuntimeError("No device connected")
        # Start streaming
        self.pipeline.start(self.config)

        self.is_speaking = False  # Flag to manage speaking state

    def list_devices(self):
        context = rs.context()
        devices = context.query_devices()
        if len(devices) == 0:
            print("No device connected")
            return False
        for device in devices:
            print(f"Device: {device.get_info(rs.camera_info.name)}")
        return True

    async def send_text(self):

        while True:
            text = await asyncio.to_thread(
                input,
                "System Information > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        while True:
            frames = await asyncio.to_thread(self.pipeline.wait_for_frames)
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert the frame to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            img.thumbnail([1024, 1024])

            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)

            mime_type = "image/jpeg"
            image_bytes = image_io.read()
            frame = {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=2,
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            #if not self.is_speaking:
            print("Listening")
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    #self.is_speaking = True
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
            

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            print("Playing")
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)
            print("Done playing")

    async def run(self):
        try:
            
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())