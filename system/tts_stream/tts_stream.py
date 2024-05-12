import asyncio
import io
import wave

_TTS_WAVE_NUM_CHANNELS = 1
_TTS_WAVE_NUM_SAMPWIDTH = 2
_TTS_WAVE_FRAMERATE = 24000
_TTS_STREAMING_HDR_CHUNK = None

class TTSStream:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.chunks = []
        self.clients: list[asyncio.Queue] = []
        self.lock = asyncio.Lock()
        self.text_queue = asyncio.Queue()
        self.text_count = 0
        self.last_received = False

    @staticmethod
    def get_streaming_header_chunk():
        global _TTS_STREAMING_HDR_CHUNK
        if _TTS_STREAMING_HDR_CHUNK is None:
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as f:
                f.setnchannels(_TTS_WAVE_NUM_CHANNELS)
                f.setsampwidth(_TTS_WAVE_NUM_SAMPWIDTH)
                f.setframerate(_TTS_WAVE_FRAMERATE)
                f.writeframes(b"")
            wav_buf.seek(0)
            _TTS_STREAMING_HDR_CHUNK = wav_buf.read()
        return _TTS_STREAMING_HDR_CHUNK

    async def add_chunk(self, chunk: bytes):
        async with self.lock:
            self.chunks.append(chunk)
            for client in self.clients:
                await client.put(chunk)

    async def stream_chunks(self):
        # Add current chunks on the client's queue
        client = None
        async with self.lock:
            # Create a queue for this client if the stream wasn't already committed
            if self.chunks is not None:
                client = asyncio.Queue()
                for chunk in self.chunks:
                    await client.put(chunk)
                self.clients.append(client)

        # If the stream was already committed, just yield the file content
        if client is None:
            with open(self.output_file, 'rb') as f:
                data = f.read()
            yield data
        # Loop until all chunks are consumed
        else:
            while True:
                chunk = await client.get()
                if chunk is None:
                    return
                yield chunk

    async def add_text(self, text: str, index: int, is_last: bool, is_single_sentence: bool):
        if index != self.text_count:
            raise Exception(f"Unexpected index {index} for text-queueing, expected {self.text_count}")
        self.text_count += 1
        if self.last_received:
            raise Exception("Last text already received, cannot queue additional text")
        await self.text_queue.put((text, is_single_sentence))
        if is_last:
            await self.text_queue.put(None)
            self.last_received = True

    async def stream_text(self):
        while True:
            text_tup = await self.text_queue.get()
            if text_tup is None:
                self.text_queue = None
                return
            text_tup: tuple[str, bool]
            yield text_tup

    async def finalize(self):
        async with self.lock:
            # Finalize the stream for all current clients
            for client in self.clients:
                await client.put(None)
            self.clients.clear()
            self.clients = None

            # Write the finalized stream to the output file
            with wave.open(self.output_file, "wb") as f:
                f.setnchannels(_TTS_WAVE_NUM_CHANNELS)
                f.setsampwidth(_TTS_WAVE_NUM_SAMPWIDTH)
                f.setframerate(_TTS_WAVE_FRAMERATE)
                for i in range(len(self.chunks)):
                    # Skip the streaming header chunk
                    # (wave rewrites its own header and the streaming-only header is hardcoded for 0-samples)
                    if i == 0:
                        continue
                    f.writeframes(self.chunks[i])
            self.chunks.clear()
            self.chunks = None
