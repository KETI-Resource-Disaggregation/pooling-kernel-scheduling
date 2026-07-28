#!/usr/bin/env python3
"""npu-proxy 테넌트 클라이언트 (벤치 겸 참조 구현, Exp_45).

폐루프·미결 depth 건 파이프라인: 커넥션 1개 위에서 depth 건을 선제출하고
응답 수신 시마다 재제출. 지연 = 제출→응답수신 (FIFO 보존이므로 순서 매칭).

usage: client.py --tenant A --data-sock <path> [--depth 4] [--dur 30]
                 [--tag A] [--out-dir logs]
"""
import argparse
import json
import os
import socket
import struct
import threading
import time
from collections import deque

import numpy as np

IN_SHAPE = (1, 3, 224, 224)


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--data-sock", required=True)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="logs")
    a = ap.parse_args()
    tag = a.tag or a.tenant
    os.makedirs(a.out_dir, exist_ok=True)
    out = open(os.path.join(a.out_dir, f"client_{tag}.jsonl"), "w", buffering=1)

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(a.data_sock)
    conn.sendall((json.dumps({"tenant": a.tenant}) + "\n").encode())

    x = np.random.randint(0, 256, IN_SHAPE, dtype=np.uint8).tobytes()
    frame = struct.pack(">I", len(x)) + x
    submits = deque()
    lat = []
    done = [0]
    stop = threading.Event()

    def receiver():
        while not stop.is_set():
            hdr = recv_exact(conn, 4)
            if hdr is None:
                return
            (n,) = struct.unpack(">I", hdr)
            if n and recv_exact(conn, n) is None:
                return
            lat.append((time.time() - submits.popleft()) * 1000)
            done[0] += 1
            sem.release()

    sem = threading.Semaphore(a.depth)
    th = threading.Thread(target=receiver, daemon=True)
    th.start()

    t0 = time.time()
    last, last_done = t0, 0
    while time.time() - t0 < a.dur:
        if not sem.acquire(timeout=0.1):
            continue
        submits.append(time.time())
        conn.sendall(frame)
        now = time.time()
        if now - last >= 1.0:
            out.write(json.dumps({"t": round(now - t0, 1),
                                  "ips": round((done[0] - last_done) / (now - last), 1)}) + "\n")
            last, last_done = now, done[0]
    # 미결 응답 회수 후 종료
    end = time.time() + 2.0
    while len(lat) < len(submits) + len(lat) - a.depth and time.time() < end:
        time.sleep(0.01)
    time.sleep(0.3)
    stop.set()
    conn.close()
    wall = time.time() - t0
    ls = sorted(lat)
    summary = {"event": "done", "tenant": a.tenant, "depth": a.depth,
               "iters": done[0], "wall_s": round(wall, 2),
               "ips": round(done[0] / wall, 1),
               "p50_ms": round(ls[len(ls) // 2], 2) if ls else None,
               "p99_ms": round(ls[int(len(ls) * 0.99)], 2) if ls else None}
    out.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
