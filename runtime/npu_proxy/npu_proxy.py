#!/usr/bin/env python3
"""npu-proxy — NPU 의 libbless (Exp_45). PE 의 유일 runner 를 소유하고
테넌트 요청을 IPC(UDS)로 받아 대행 실행, run() 호출 비율을 게이트로 배분한다.

구조 대응 (GPU ↔ NPU):
  libbless(주입)      ↔ npu-proxy(독립 프로세스, PE 소유)
  bless-1.sock        ↔ <sock-dir>/<tenant>.sock  (같은 명령 계열)
  stderr→테넌트 로그   ↔ <log-dir>/<tenant>.log    (time_stats 응답 형식 동일)
  controller feeder   ↔ 동일 feeder 재사용 (라우트/tick/회계 무변경)

데이터 평면 (UDS stream, <data-sock>):
  handshake: JSON 1줄 {"tenant": "<id>"}\n   (미등록 테넌트면 자동 등록)
  요청     : 4B big-endian 길이 + 입력 텐서 bytes (모델 입력 shape 고정)
  응답     : 4B big-endian 길이 + 출력 텐서 bytes (테넌트별 FIFO 순서 보존)

제어 평면 (테넌트별 UDS DGRAM, libbless 계열 — feeder 계약 Exp_16/26):
  time_mode <0|1> / time_credit <us|-1> / time_add <us>
  time_stats → <log-dir>/<tenant>.log 에 "... total=<us> kernels=<n>" 발화
관리 소켓 (<admin-sock>, DGRAM):
  register <t> / unregister <t> / npu_stats (→ <log-dir>/proxy.log 에 JSON)

스케줄링 = gate_core.GateCore (Exp_44 확정본: deficit 선택·strict·실측 차감).
"""
import argparse
import json
import os
import socket
import struct
import threading
import time

import numpy as np

from gate_core import GateCore

IN_SHAPE = (1, 3, 224, 224)     # ResNet50 INT8 (Exp_43/44 경로)
IN_BYTES = int(np.prod(IN_SHAPE))


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class Proxy:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.sock_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        self.logs = {}                    # tenant -> file
        self.ctrl_socks = {}              # tenant -> socket (닫기용)
        self.proxy_log = open(os.path.join(args.log_dir, "proxy.log"), "a",
                              buffering=1)
        # runner (PE 유일 소유). --enf 가 있으면 ENF 파일 직접 로드 —
        # 컨테이너에서 furiosa-models(무거운 의존) 없이 runtime 만으로 구동 (Exp_47).
        from furiosa.runtime.sync import create_runner
        t0 = time.time()
        if args.enf:
            src = args.enf
        else:
            from furiosa.models.vision import ResNet50
            src = ResNet50().model_source(num_pe=1)
        self.sess = create_runner(src, device=args.pe)
        self._plog({"event": "runner_up", "pe": args.pe,
                    "model": args.model_name,
                    "enf": args.enf or "furiosa-models(resnet50 1pe)",
                    "create_s": round(time.time() - t0, 2)})
        self.core = GateCore(self._run_req)
        # sanity gate (Exp_1 계열)
        x = np.random.randint(0, 256, IN_SHAPE, dtype=np.uint8)
        warm = []
        for _ in range(5):
            s = time.time()
            self.sess.run([x])
            warm.append((time.time() - s) * 1000)
        self._plog({"event": "warmup", "warm_ms": [round(w, 2) for w in warm]})
        if warm[-1] > 100:
            self._plog({"event": "SANITY_ABORT"})
            raise SystemExit(2)

    def _plog(self, obj):
        self.proxy_log.write(json.dumps(obj) + "\n")

    def _run_req(self, req):
        x = np.frombuffer(req["body"], dtype=np.uint8).reshape(IN_SHAPE)
        req["out"] = np.asarray(self.sess.run([x])[0])   # sync runner = ndarray 반환

    # ---- 테넌트 등록 (제어 소켓 + 로그 개설) ----
    def ensure_tenant(self, name):
        if name in self.logs:
            return
        self.core.register(name)
        self.logs[name] = open(os.path.join(self.args.log_dir, f"{name}.log"),
                               "a", buffering=1)
        path = os.path.join(self.args.sock_dir, f"{name}.sock")
        try:
            os.unlink(path)
        except OSError:
            pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.bind(path)
        self.ctrl_socks[name] = s
        threading.Thread(target=self._ctrl_loop, args=(name, s),
                         daemon=True).start()
        self._plog({"event": "tenant_up", "tenant": name, "ctrl_sock": path})

    def drop_tenant(self, name):
        for req in self.core.unregister(name):
            self._reply_err(req)
        s = self.ctrl_socks.pop(name, None)
        if s:
            s.close()
            try:
                os.unlink(os.path.join(self.args.sock_dir, f"{name}.sock"))
            except OSError:
                pass
        f = self.logs.pop(name, None)
        if f:
            f.close()
        self._plog({"event": "tenant_down", "tenant": name})

    # ---- 제어 평면 ----
    def _ctrl_loop(self, name, s):
        while True:
            try:
                data, _ = s.recvfrom(256)
            except OSError:
                return
            parts = data.decode(errors="replace").split()
            if not parts:
                continue
            cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
            try:
                resp = self.core.cmd(name, cmd, arg)
            except (ValueError, TypeError) as e:
                self.logs[name].write(f"[npu-proxy] bad cmd {parts}: {e}\n")
                continue
            if resp is not None and name in self.logs:
                self.logs[name].write(resp + "\n")

    def _admin_loop(self, s):
        while True:
            data, _ = s.recvfrom(256)
            parts = data.decode(errors="replace").split()
            if not parts:
                continue
            if parts[0] == "register" and len(parts) > 1:
                self.ensure_tenant(parts[1])
            elif parts[0] == "unregister" and len(parts) > 1:
                self.drop_tenant(parts[1])
            elif parts[0] == "npu_stats":
                self._plog({"event": "npu_stats", "pe": self.args.pe,
                            "model": self.args.model_name,
                            "tenants": self.core.stats()})

    # ---- 데이터 평면 ----
    def _reply_err(self, req):
        try:
            req["conn"].sendall(struct.pack(">I", 0))
        except OSError:
            pass

    def _conn_loop(self, conn):
        # 핸드셰이크는 바이트 단위로 개행까지만 읽는다 — makefile readline 은
        # 선독 버퍼가 첫 프레임을 삼켜 프레이밍이 깨진다 (간헐 0건 처리의 근인)
        line = b""
        while not line.endswith(b"\n"):
            c = conn.recv(1)
            if not c:
                conn.close()
                return
            line += c
        hello = json.loads(line)
        name = hello["tenant"]
        self.ensure_tenant(name)
        while True:
            hdr = recv_exact(conn, 4)
            if hdr is None:
                break
            (n,) = struct.unpack(">I", hdr)
            body = recv_exact(conn, n)
            if body is None or n != IN_BYTES:
                break
            self.core.submit(name, {"conn": conn, "body": body})
        conn.close()

    # ---- 스케줄러 루프 ----
    def _sched_loop(self):
        while True:
            try:
                r = self.core.step()
            except Exception as e:          # runner 오류 — 스케줄러는 죽지 않는다
                self._plog({"event": "run_error", "err": f"{type(e).__name__}: {e}"})
                continue
            if r is None:
                time.sleep(0.0002)
                continue
            _, req, _ = r
            out = req["out"].tobytes()
            try:
                req["conn"].sendall(struct.pack(">I", len(out)) + out)
            except OSError:
                pass                       # 클라이언트 이탈 — 결과 폐기

    def serve(self):
        threading.Thread(target=self._sched_loop, daemon=True).start()
        adm = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            os.unlink(self.args.admin_sock)
        except OSError:
            pass
        adm.bind(self.args.admin_sock)
        threading.Thread(target=self._admin_loop, args=(adm,),
                         daemon=True).start()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.unlink(self.args.data_sock)
        except OSError:
            pass
        srv.bind(self.args.data_sock)
        srv.listen(16)
        self._plog({"event": "serving", "data_sock": self.args.data_sock,
                    "admin_sock": self.args.admin_sock})
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self._conn_loop, args=(conn,),
                             daemon=True).start()


def main():
    ap = argparse.ArgumentParser(prog="npu-proxy")
    ap.add_argument("--pe", default="npu0pe0")
    ap.add_argument("--enf", default="",
                    help="컴파일된 ENF 파일 경로 (지정 시 furiosa-models 불필요)")
    ap.add_argument("--model-name", default="resnet50_int8",
                    help="로그·npu_stats 표기용 모델명")
    ap.add_argument("--data-sock", required=True)
    ap.add_argument("--admin-sock", required=True)
    ap.add_argument("--sock-dir", required=True, help="테넌트별 제어 소켓 디렉토리")
    ap.add_argument("--log-dir", required=True, help="테넌트별 로그 디렉토리")
    Proxy(ap.parse_args()).serve()


if __name__ == "__main__":
    main()
