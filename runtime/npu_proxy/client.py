#!/usr/bin/env python3
"""npu-proxy 테넌트 클라이언트 (벤치 겸 참조 구현, Exp_45 → Exp_55 재접속).

폐루프·미결 depth 건 파이프라인: 커넥션 1개 위에서 depth 건을 선제출하고
응답 수신 시마다 재제출. 지연 = 제출→응답수신 (FIFO 보존이므로 순서 매칭).

★재접속 정책 (Exp_55 — 프록시 재기동 내성):
  · 연결 끊김(EOF/OSError) 감지 → 지수 백오프(0.2s→최대 3.2s) 재연결, 핸드셰이크
    (테넌트 ID 유지) 재수행. 프록시는 같은 테넌트의 재등록을 수용한다
    (ensure_tenant 가 멱등 — 소켓/로그 재사용).
  · **재연결 중 요청은 큐잉하지 않는다**: 진행 중이던 미결 요청은 프록시가
    무상태 대행이라 유실이 확정이므로, 세마포어를 반납해 새 요청으로 대체한다
    (재전송하지 않음 — 중복 추론 방지). 유실 건수는 lost 로 계수·보고.
  · 재연결 실패가 계속되면 요청 루프는 계속 시도하되 진행하지 않는다(정직한 정체).

usage: client.py --tenant A --data-sock <path> [--depth 4] [--dur 30]
                 [--tag A] [--out-dir logs] [--no-reconnect]
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
BACKOFF_START = 0.2
BACKOFF_MAX = 3.2


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        try:
            chunk = conn.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


class Connection:
    """재접속 가능한 프록시 연결 — 소켓 교체를 한 곳에서 관리(스레드 안전)."""

    def __init__(self, path, tenant, reconnect=True):
        self.path = path
        self.tenant = tenant
        self.reconnect_enabled = reconnect
        self.lock = threading.Lock()
        self.sock = None
        self.epoch = 0          # 재연결 세대 — 수신부가 낡은 소켓을 구분
        self.reconnects = 0
        self.connect()

    def connect(self):
        """새 소켓으로 연결 + 핸드셰이크. 실패 시 예외."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.path)
        s.sendall((json.dumps({"tenant": self.tenant}) + "\n").encode())
        with self.lock:
            old = self.sock
            self.sock = s
            self.epoch += 1
        if old is not None:
            try:
                old.close()
            except OSError:
                pass

    def reconnect_loop(self, stop, my_epoch):
        """my_epoch 세대의 끊김을 보고 재연결. 이미 다른 스레드가 갱신했으면 무시."""
        if not self.reconnect_enabled:
            return False
        with self.lock:
            if self.epoch != my_epoch:
                return True     # 다른 경로가 이미 재연결함
        delay = BACKOFF_START
        while not stop.is_set():
            try:
                self.connect()
                self.reconnects += 1
                return True
            except OSError:
                time.sleep(delay)
                delay = min(delay * 2, BACKOFF_MAX)
        return False

    def current(self):
        with self.lock:
            return self.sock, self.epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--data-sock", required=True)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--no-reconnect", action="store_true",
                    help="재접속 비활성 (Exp_53 이전 동작 재현용)")
    a = ap.parse_args()
    tag = a.tag or a.tenant
    os.makedirs(a.out_dir, exist_ok=True)
    out = open(os.path.join(a.out_dir, f"client_{tag}.jsonl"), "w", buffering=1)

    conn = Connection(a.data_sock, a.tenant, reconnect=not a.no_reconnect)
    x = np.random.randint(0, 256, IN_SHAPE, dtype=np.uint8).tobytes()
    frame = struct.pack(">I", len(x)) + x
    submits = deque()
    lat = []
    done = [0]
    lost = [0]
    stop = threading.Event()
    sem = threading.Semaphore(a.depth)

    def drop_inflight():
        """끊김 시 미결 요청 폐기 — 세마포어 반납(재전송 안 함)."""
        n = 0
        while submits:
            submits.popleft()
            sem.release()
            n += 1
        lost[0] += n

    def receiver():
        while not stop.is_set():
            s, ep = conn.current()
            hdr = recv_exact(s, 4)
            if hdr is None:
                if stop.is_set():
                    return
                drop_inflight()
                if not conn.reconnect_loop(stop, ep):
                    return
                continue
            (n,) = struct.unpack(">I", hdr)
            if n and recv_exact(s, n) is None:
                drop_inflight()
                if not conn.reconnect_loop(stop, ep):
                    return
                continue
            if submits:
                lat.append((time.time() - submits.popleft()) * 1000)
            done[0] += 1
            sem.release()

    threading.Thread(target=receiver, daemon=True).start()

    t0 = time.time()
    last, last_done = t0, 0
    while time.time() - t0 < a.dur:
        if not sem.acquire(timeout=0.1):
            continue
        s, ep = conn.current()
        submits.append(time.time())
        try:
            s.sendall(frame)
        except OSError:
            drop_inflight()
            conn.reconnect_loop(stop, ep)
            continue
        now = time.time()
        if now - last >= 1.0:
            out.write(json.dumps({"t": round(now - t0, 1),
                                  "ips": round((done[0] - last_done) / (now - last), 1),
                                  "reconnects": conn.reconnects}) + "\n")
            last, last_done = now, done[0]
    stop.set()
    try:
        conn.current()[0].close()
    except OSError:
        pass
    wall = time.time() - t0
    ls = sorted(lat)
    summary = {"event": "done", "tenant": a.tenant, "depth": a.depth,
               "iters": done[0], "wall_s": round(wall, 2),
               "ips": round(done[0] / wall, 1),
               "reconnects": conn.reconnects, "lost_inflight": lost[0],
               "p50_ms": round(ls[len(ls) // 2], 2) if ls else None,
               "p99_ms": round(ls[int(len(ls) * 0.99)], 2) if ls else None}
    out.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
