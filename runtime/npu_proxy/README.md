# npu-proxy — NPU 의 libbless (Exp_44 PoC → Exp_45 정본 승격)

PE 는 배타 단위라(동일 PE 이중 runner 즉시 거부 — Exp_36/43) GPU 식 인터셉트가
불가하다. npu-proxy 가 PE 의 **유일 runner** 를 소유하고 테넌트 요청을 대행 실행하며
run() 호출 비율을 게이트로 배분한다. **NPU 는 시간 축 단독** — 공간(s) 개념 미적용.

## 구조 대응 (GPU ↔ NPU)

| | GPU | NPU |
|---|---|---|
| data plane | libbless (LD_PRELOAD 주입) | npu-proxy (독립 프로세스, PE 소유) |
| 제어 채널 | `bless-<pid>.sock` | `<sock-dir>/<tenant>.sock` (명령 계열 동일) |
| control plane | controller feeder (tick 10ms) | **동일 feeder 재사용** (`resource: npu` 태그만) |
| 회계 | charged_us (time_stats) | 동일 — 추론 실측시간 누적 |
| 투명성 | zero-modification | ★앱이 client 프로토콜 경유 필요 |
| 모델 | 테넌트별 자유 | ★runner 당 1모델 — **같은 모델 테넌트 한정** |

## 실행

```bash
python3 npu_proxy.py --pe npu0pe0 \
  --data-sock /run/prism/npu-data.sock --admin-sock /run/prism/npu-admin.sock \
  --sock-dir /run/prism/socks --log-dir /run/prism/logs
```

기동 시 sanity gate(워밍업 >100ms 중단, Exp_1 계열).

## 데이터 평면 (UDS stream, `--data-sock`)

1. handshake: JSON 1줄 `{"tenant": "<id>"}\n` — 미등록 테넌트는 자동 등록
2. 요청: `4B big-endian 길이` + 입력 텐서 bytes (ResNet50 INT8: (1,3,224,224) uint8 = 150528B)
3. 응답: `4B big-endian 길이` + 출력 텐서 bytes — 테넌트별 FIFO 순서 보존
4. 파이프라인: depth≥2 권장 (depth1 은 재제출 갭으로 처리율 −15%대 — Exp_44/45 실측)

참조 클라이언트: `client.py` (폐루프 depth 파이프라인 + 1초 ips 로그).
★핸드셰이크는 서버가 개행까지 바이트 단위로 읽는다 — buffered readline 은 첫
프레임을 선독해 프레이밍이 깨진다 (Exp_45 에서 실측한 간헐 결함).

## 제어 평면 (테넌트별 UDS DGRAM — feeder 계약 Exp_16/26 그대로)

| 명령 | 의미 |
|---|---|
| `time_mode <0\|1>` | 게이트 무장/해제 (1 = credit 기반 실행 자격) |
| `time_credit <us>` | 잔고 설정. `0`=차단 시작, `-1`=unlimited |
| `time_add <us>` | 예산 주입 (feeder 가 tick 마다) |
| `time_stats` | `<log-dir>/<tenant>.log` 에 `... total=<charged_us> kernels=<완료수>` 발화 |

응답 형식은 libbless 와 동일 정규식(`total=(-?\d+) kernels=(\d+)`)으로 파싱된다 —
feeder `read_time_stats()` 무변경 재사용.

관리 소켓(`--admin-sock`, DGRAM): `register <t>` / `unregister <t>` /
`npu_stats`(→ `<log-dir>/proxy.log` 에 JSON).

## 스케줄링 (gate_core.py — Exp_44 확정본)

- 선택 = credit>0 이고 대기 요청 있는 테넌트 중 **min(charged/granted)**
  (stride/deficit — granted 는 누적 주입 예산이라 feeder 비율이 그대로 비례가 된다)
- strict: 전원 credit 소진이면 PE 유휴 (봉투 격리 우선 — libbless 와 동일 의미론)
- 실행 후 실측 시간(us)을 charged 누적·credit 차감. credit 상한 100ms 클램프.

★Exp_44 오답 2건 재발 방지 (test_16 이 회귀 감시):
1. **max-credit 선택 금지** — credit 소비율 균등화로 어떤 목표든 5:5 붕괴.
2. **depth1 + 작업보존 금지** — 재제출 갭이 강제 교대를 만들어 5:5 붕괴.
   본 구현은 작업보존 폴백 자체가 없다(strict).

## controller 연동 (Exp_45)

`POST /feeder/register` 에 `resource: "npu"` (기본 `"gpu"`) — 채널·명령·회계가
동일하므로 feeder 동작은 타입 무관, 태그는 관측/문서용. E2E 실측(Exp_45):
7:3→0.7004, 런타임 3:7 전환→0.3000, 이탈 재분배→0.9942, 합류 복귀→0.7030.
IPC(UDS) 전송 오버헤드 −0.8% (150KB/req, depth4).
