#!/usr/bin/env python3
import os, sys, socket, time, errno

def sock(pid): return f"/tmp/bless-{pid}.sock"

def send(pid, msg, retries=30, delay=0.2):
    path = sock(pid)
    data = msg.encode()
    for i in range(retries):
        try:
            s=socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            # connect 대신 sendto로 직접 전송 (서버가 늦게 bind여도 덜 깨짐)
            s.sendto(data, path)
            s.close()
            return True
        except OSError as e:
            # 소켓 파일이 아직 없거나 (ENOENT), 혹은 Refused면 재시도
            if e.errno in (errno.ENOENT, errno.ECONNREFUSED):
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"cannot send to {path}: {msg}")

if __name__=="__main__":
    if len(sys.argv)<3:
        print("usage: blessctl <pid> <limited|unlimited|barrier>")
        sys.exit(1)
    ok = send(int(sys.argv[1]), sys.argv[2])
    if not ok: sys.exit(2)
