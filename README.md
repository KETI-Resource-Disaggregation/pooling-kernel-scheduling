# Pooling Kernel Scheduling

GPU 멀티테넌트 환경을 위한 커널 스케줄링 시스템

## 개요

SPARK (Scheduler with Priority-aware Auto Planner) 기반의 GPU 커널 스케줄링 시스템입니다.
딥러닝 워크로드의 GPU 공유를 위한 시간/공간 분할 스케줄링을 제공합니다.

## 아키텍처

```
                    ┌─────────────────────────────┐
                    │   pooling-workload-profiler │
                    │      (Port 7070)            │
                    └──────────────┬──────────────┘
                                   │ REST API
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 pooling-kernel-scheduling                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │  scheduler  │  │  sched_api  │  │ colocation_scheduler │ │
│  │  (Port 6060)│  │  (Port 8080)│  │       (C++)          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────────────────┘ │
│         │                │                                   │
│         └────────┬───────┘                                   │
│                  │ Unix Socket                               │
│                  ▼                                           │
│         ┌───────────────┐                                    │
│         │    libbless   │ ← CUDA Interposition               │
│         │ (Time Credit) │                                    │
│         └───────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │     GPU      │
            └──────────────┘
```

## 주요 기능

### 1. 시간 기반 크레딧 시스템 (`libbless/`)
- 커널 개수 대신 실제 GPU 시간(μs)으로 크레딧 관리
- 공정한 GPU 시간 배분
- CUDA 함수 인터포지션

### 2. Co-location 스케줄러 (`colocation_scheduler/`)
- Orion 연구 기반 간섭 최소화 배치
- 커널 타입 기반 최적 조합 권장
- SM 파티션 권장

### 3. 중앙 스케줄러 (`controller/scheduler.py`)
- Squad 기반 크레딧 분배
- 우선순위 지원 (HIGH/MED/LOW)
- Boost/Solo 모드 지원
- 자동 플래닝 (Lexicographic, Waterfilling)

### 4. HTTP API (`controller/sched_api.py`)
- 테넌트 런칭/관리
- 스케줄러 제어
- 실시간 모니터링

## 시스템 요구사항

- NVIDIA GPU (Compute Capability 7.0+)
- CUDA 12.0+
- Python 3.8+
- PyTorch 2.0+ (워크로드 실행 시)

## 빌드

```bash
# 전체 빌드
make

# libbless만 빌드
cd libbless && make

# colocation_scheduler 빌드
make colocation_scheduler
```

## 실행

### 1. 스케줄러 시작

```bash
# 스케줄러 (필수)
python3 controller/scheduler.py

# API 서버 (선택)
python3 controller/sched_api.py
```

### 2. 워크로드 실행

```bash
# libbless와 함께 워크로드 실행
LD_PRELOAD=libbless/libbless.so \
BLESS_TENANT=A \
BLESS_MASTER=/tmp/bless-master.sock \
python3 controller/run_multi.py --domain nlp --model gpt2 --steps 200
```

### 3. 스케줄러 제어

```bash
# CLI로 제어
python3 controller/schedctl.py set_equal 1
python3 controller/schedctl.py set_weights A:50,B:30,C:20
python3 controller/schedctl.py boost A

# 테넌트 제어
python3 controller/blessctl.py --list
python3 controller/blessctl.py -t A set_share 250
```

## 폴더 구조

```
pooling-kernel-scheduling/
├── libbless/                    # CUDA 인터포저 라이브러리
│   ├── libbless.cpp            # 시간 기반 크레딧 시스템
│   ├── context_manager.hpp     # 컨텍스트 관리
│   ├── routing.hpp             # 라우팅 헤더
│   └── Makefile
├── colocation_scheduler/        # Co-location 스케줄러
│   ├── colocation_scheduler.h
│   ├── colocation_scheduler.cpp
│   └── colocation_demo.cu
├── controller/                  # Python 스케줄러 인프라
│   ├── scheduler.py            # 중앙 스케줄러
│   ├── sched_api.py            # HTTP API 서버
│   ├── schedctl.py             # 스케줄러 CLI
│   ├── blessctl.py             # 테넌트 제어 CLI
│   ├── profiler_client.py      # Profiler API 클라이언트
│   ├── run_multi.py            # 워크로드 러너
│   └── run_infer.py            # 인퍼런스 러너
└── docs/                        # 문서
```

## 환경 변수

### 스케줄러 설정
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BLESS_MASTER` | `/tmp/bless-master.sock` | 마스터 소켓 경로 |
| `SQUAD` | 10000 | Squad 크기 |
| `TICK_S` | 0.006 | 크레딧 틱 간격 (초) |
| `CREDIT_PER_TICK` | 524288 | 틱당 크레딧 |
| `EQUAL_SHARE` | 1 | 균등 분배 (1) / 가중치 (0) |
| `GOAL_POLICY` | lexi | 플래닝 정책 (lexi/wf) |

### API 서버 설정
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `API_HOST` | 0.0.0.0 | API 호스트 |
| `API_PORT` | 8080 | API 포트 |
| `API_KEY` | (없음) | 인증 키 |

### Profiler 연동
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PROFILER_URL` | http://localhost:7070 | Profiler API URL |

## API 엔드포인트

### 스케줄러 API (Port 6060)
- `GET /stats` - 스케줄러 상태
- `POST /telemetry` - 테넌트 메트릭 수신
- `POST /goal`, `/goals` - SLA 목표 설정
- `POST /priority` - 우선순위 설정

### 관리 API (Port 8080)
- `POST /launch` - 테넌트 시작
- `POST /stop`, `/kill` - 테넌트 중지
- `GET /tenants` - 테넌트 목록
- `POST /set_equal`, `/set_weights` - 스케줄링 설정
- `POST /boost`, `/boost_off` - 부스트 제어

## pooling-workload-profiler 연동

```python
from controller.profiler_client import ProfilerClient, should_colocate

# 프로파일러 연결
client = ProfilerClient("http://localhost:7070")

# 워크로드 프로파일 조회
profile = client.get_profile("resnet50")
print(f"Type: {profile.kernel_type.name}")

# Co-location 확인
ok, score = should_colocate("resnet50", "bert")
print(f"Co-locate: {ok}, Score: {score:.2f}")
```

## 참고 문헌

- Orion: Interference-aware, Interference-free Multi-tenant GPU Scheduling (OSDI'22)
