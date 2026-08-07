#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""say.py — 发送文本到 HiggsTTS TCP 服务，流式接收并实时播放语音。

用法:
    python say.py "要说的话"
    python say.py --port 9989 --temp 0.9 --stream "要说的话"

协议 (流式):
    请求: 4字节文本长度(BE) + 4字节温度(BE float) + UTF-8 文本
    响应: 帧序列 [1B type][4B payload bytes BE][payload]
          type=1: float32 PCM @ 24kHz
          type=2: 结束
          type=3: UTF-8 错误信息

不传 --stream 时使用旧协议（一次性返回全部 PCM）。
"""
import argparse
import array
import io
import queue
import socket
import struct
import sys
import threading
import wave

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9988
SAMPLE_RATE = 24000


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接被服务器关闭，收到的数据不完整")
        buf.extend(chunk)
    return bytes(buf)


def f32_to_wav_bytes(pcm_f32):
    """float32 PCM → 内存 WAV bytes（16-bit mono）"""
    int16 = array.array("h")
    for s in pcm_f32:
        v = int(max(-1.0, min(1.0, s)) * 32767.0)
        int16.append(v)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


def play_pcm(pcm_f32):
    """播放 float32 PCM（同步，阻塞到播放完）"""
    import winsound
    winsound.PlaySound(f32_to_wav_bytes(pcm_f32), winsound.SND_MEMORY)


# ── streaming playback ────────────────────────────────────────────────────────
def streaming_recv_and_play(sock, t_request):
    """流式协议：逐帧接收 PCM，pyaudio 实时 gapless 播放。"""

    import collections
    import pyaudio
    import time

    p = pyaudio.PyAudio()
    error_msg = [None]
    sample_buf = collections.deque()
    stream_done = threading.Event()
    t_first_chunk = [0.0]
    t_first_sound = [0.0]
    chunk_received = [False]
    sound_played = [False]
    total_samples = [0]

    def _callback(in_data, frame_count, time_info, status):
        """pyaudio 回调：从 sample_buf 取 frame_count 个采样点"""
        out = []
        has_samples = False
        for _ in range(frame_count):
            if sample_buf:
                out.append(sample_buf.popleft())
                has_samples = True
            else:
                out.append(0.0)
        if has_samples and not sound_played[0]:
            t_first_sound[0] = time.perf_counter()
            sound_played[0] = True
        if not sample_buf and stream_done.is_set():
            return (array.array("f", out).tobytes(), pyaudio.paComplete)
        return (array.array("f", out).tobytes(), pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=960,
        stream_callback=_callback,
    )

    try:
        while True:
            type_byte = recv_exact(sock, 1)
            frame_type = type_byte[0]
            payload_len = struct.unpack(">I", recv_exact(sock, 4))[0]

            if frame_type == 1:  # PCM chunk
                if payload_len == 0:
                    continue
                pcm_data = recv_exact(sock, payload_len)
                if not chunk_received[0]:
                    t_first_chunk[0] = time.perf_counter()
                    chunk_received[0] = True
                n_samples = payload_len // 4
                total_samples[0] += n_samples
                sample_buf.extend(struct.unpack(f"<{n_samples}f", pcm_data))

            elif frame_type == 2:  # end
                t_end = time.perf_counter()
                total_sec = total_samples[0] / SAMPLE_RATE
                synth_ms = (t_end - t_request) * 1000
                rtf = synth_ms / 1000 / total_sec if total_sec > 0 else 0
                print(f"[latency] 合成完成:   {synth_ms:.0f} ms  (音频 {total_sec:.1f}s, RTF {rtf:.3f})")
                break

            elif frame_type == 3:  # error
                if payload_len > 0:
                    error_msg[0] = recv_exact(sock, payload_len).decode("utf-8", errors="replace")
                break

            else:
                print(f"未知帧类型: {frame_type}", file=sys.stderr)
                break
    finally:
        stream_done.set()
        while stream.is_active():
            time.sleep(0.05)
        stream.stop_stream()
        stream.close()
        p.terminate()

    if chunk_received[0]:
        ttfb = (t_first_chunk[0] - t_request) * 1000
        print(f"[latency] 首字节到达: {ttfb:.0f} ms")
    if sound_played[0]:
        ttfs = (t_first_sound[0] - t_request) * 1000
        print(f"[latency] 开始播音:   {ttfs:.0f} ms")

    if error_msg[0]:
        sys.exit(f"服务端合成失败: {error_msg[0]}")


# ── legacy (non-streaming) ────────────────────────────────────────────────────
def legacy_recv_and_play(sock):
    """旧协议：一次性接收全部 PCM 后播放"""
    header = recv_exact(sock, 4)
    n_samples = struct.unpack(">i", header)[0]
    if n_samples < 0:
        sys.exit(f"服务端合成失败（返回 {n_samples}）")

    pcm_data = recv_exact(sock, n_samples * 4)
    pcm = struct.unpack(f"<{n_samples}f", pcm_data)
    play_pcm(pcm)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="发送文本到 HiggsTTS 服务并播放语音")
    parser.add_argument("text", nargs="*", help="要说的话（多段自动拼接）")
    parser.add_argument("--host", default=DEFAULT_HOST, help="服务地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="服务端口，默认 9989")
    parser.add_argument("--temp", type=float, default=0.9, help="temperature，默认 0.9")
    parser.add_argument("--no-stream", action="store_true", help="使用旧协议（一次性返回全部 PCM 再播放）")
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    if not text:
        parser.error('请提供要说的话：python say.py "要说的话"')

    text_bytes = text.encode("utf-8")

    with socket.create_connection((args.host, args.port), timeout=120) as s:
        s.sendall(struct.pack(">I", len(text_bytes)))
        s.sendall(struct.pack(">f", args.temp))
        s.sendall(text_bytes)

        import time
        t_request = time.perf_counter()

        if args.no_stream:
            legacy_recv_and_play(s)
        else:
            streaming_recv_and_play(s, t_request)


if __name__ == "__main__":
    main()
