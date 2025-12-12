# Soft SM Partitioning + Work-conserving 설계 문서

## 1. 목표

### 1.1 요구사항
- 워크로드별 기본 SM 할당
- 요청량/부하에 따른 동적 SM 재할당
- Work-conserving: 유휴 SM 흡수 (한 워크로드 종료 시 다른 워크로드가 SM 활용)

### 1.2 제약사항 (CUDA 한계)
- `cuCtxCreate_v3`로 설정한 SM affinity는 메모리 할당 후 변경 불가
- MPS 환경에서만 SM 파티셔닝 동작
- 런타임 중 SM 개수 직접 변경 불가

---

## 2. 설계 접근법

### 2.1 하이브리드 시공간 분할

```
┌─────────────────────────────────────────────────────────────────┐
│                    시공간 분할 스케줄링                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │    공간 분할 (SM)    │      │    시간 분할 (Credit)│          │
│  │                     │      │                     │          │
│  │ • 초기 SM% 할당     │  +   │ • 시간 크레딧 관리   │          │
│  │ • 프로세스별 고정   │      │ • 동적 재분배       │          │
│  │ • MPS 기반         │      │ • Work-conserving   │          │
│  └─────────────────────┘      └─────────────────────┘          │
│                                                                  │
│                    ═══════════════════════                       │
│                           결합                                   │
│                    ═══════════════════════                       │
│                                                                  │
│  SM 비율이 낮은 워크로드는 더 많은 시간 크레딧을 받아 보상        │
│  유휴 워크로드의 크레딧을 활성 워크로드로 전이 (Work-conserving)  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 핵심 아이디어

1. **초기 SM 할당**: 프로세스 시작 시 커널 타입 기반 SM 비율 결정
2. **시간 보상**: SM 비율이 낮으면 시간 크레딧을 더 많이 할당
3. **Work-conserving**: 유휴 워크로드의 크레딧을 활성 워크로드로 전이

---

## 3. 구현 설계

### 3.1 SM Manager (새 컴포넌트)

```python
class SMManager:
    """
    워크로드별 SM 할당 관리
    """
    def __init__(self, total_sms: int = 84):
        self.total_sms = total_sms
        self.allocations = {}  # tenant_id -> SMAllocation

    def allocate(self, tenant_id: str, kernel_type: str, priority: str) -> int:
        """
        커널 타입과 우선순위 기반 SM 할당
        Returns: 할당된 SM 개수
        """
        pass

    def compute_time_compensation(self, tenant_id: str) -> float:
        """
        SM 비율에 따른 시간 보상 계수 계산
        SM이 적으면 더 많은 시간 크레딧 할당
        """
        pass

    def redistribute_idle(self, idle_tenant: str, active_tenants: List[str]):
        """
        유휴 워크로드의 자원을 활성 워크로드로 재분배
        """
        pass
```

### 3.2 확장된 스케줄러 로직

```
워크로드 시작 시:
1. profiler에서 커널 타입 조회
2. SM Manager에서 초기 SM% 결정
3. libbless에 SM% 전달 (BLESS_LIMIT_PCT 환경변수)
4. 시간 보상 계수 적용하여 기본 크레딧 할당

런타임 중:
1. 텔레메트리 수신 (QPS, latency)
2. 유휴 감지 시 Work-conserving 트리거
3. 활성 워크로드에 추가 크레딧 할당

워크로드 종료 시:
1. SM 할당 해제
2. 남은 크레딧 재분배
```

### 3.3 Work-conserving 메커니즘

```
┌─────────────────────────────────────────────────────────────────┐
│                    Work-conserving Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  워크로드 A (SM 60%)     워크로드 B (SM 40%)                     │
│  ────────────────────    ────────────────────                    │
│                                                                  │
│  [Active]                [Idle - 작업 완료]                      │
│     │                         │                                  │
│     │    ◀───── 크레딧 전이 ─────                               │
│     │                                                            │
│  [Boosted]               [Waiting]                               │
│  + 추가 크레딧                                                   │
│  + SM 100% 활용 가능 (time slice 내)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. API 설계

### 4.1 SM Manager API (scheduler → libbless)

```bash
# 초기 SM 비율 설정 (프로세스 시작 전)
BLESS_LIMIT_PCT=60

# 런타임 SM 재설정 (할당 전에만 가능)
echo "set_limit_pct 60" | nc -U /tmp/bless-{pid}.sock

# SM 개수 직접 설정
echo "reconf_sm 50" | nc -U /tmp/bless-{pid}.sock
```

### 4.2 확장된 스케줄러 API

```bash
# SM 할당 조회
GET /sm_allocations

# SM 할당 설정 (새 워크로드)
POST /sm_allocate
{
    "tenant": "A",
    "sm_pct": 60,
    "kernel_type": "COMPUTE_BOUND"
}

# Work-conserving 상태
GET /work_conserving_status
```

### 4.3 Profiler 연동 API

```bash
# SM 권장 비율 조회
GET /colocation?a=resnet50&b=bert

Response:
{
    "sm_partition": {
        "workload_a": 50,  # SM 개수
        "workload_b": 34
    },
    "time_compensation": {
        "workload_a": 1.0,
        "workload_b": 1.47  # SM이 적으므로 시간 보상
    }
}
```

---

## 5. 구현 단계

### Phase 1: SM Manager 기본 구현 ✅ 완료
- [x] `sm_manager.py` 생성
- [x] 커널 타입 기반 SM 할당 로직
- [x] 시간 보상 계수 계산

### Phase 2: 스케줄러 통합 ✅ 완료
- [x] `scheduling_coordinator.py` 생성 (SM Manager 연동)
- [x] 워크로드 시작 시 SM 할당
- [x] 텔레메트리 기반 유휴 감지

### Phase 3: Work-conserving ✅ 완료
- [x] 유휴 감지 로직
- [x] 크레딧 재분배 메커니즘
- [x] Boost 연동 (최대 2.0x)

### Phase 4: Profiler 연동 ✅ 완료
- [x] SM 권장 비율 API (profiler_client.py)
- [x] 시간 보상 계수 API
- [x] 커널 타입 자동 조회

---

## 6. 예상 시나리오

### 시나리오 1: ResNet50 + BERT 동시 실행

```
초기 상태:
- ResNet50: COMPUTE_BOUND → SM 60%, 크레딧 1.0x
- BERT: MIXED → SM 40%, 크레딧 1.5x (시간 보상)

실행 중:
- 두 워크로드 모두 자신의 SM 내에서 실행
- BERT는 SM이 적지만 크레딧이 많아 시간적으로 보상

ResNet50 종료 시:
- BERT에 추가 크레딧 전이 (Work-conserving)
- BERT가 남은 작업을 더 빠르게 처리
```

### 시나리오 2: 우선순위 기반 재조정

```
HIGH 우선순위 워크로드 도착:
- 기존 LOW 워크로드의 크레딧 일부 회수
- HIGH 워크로드에 더 많은 크레딧 할당
- SM 비율은 유지 (이미 할당됨)
```

---

## 7. 제약사항 및 향후 개선

### 현재 제약
1. SM 파티션은 프로세스 시작 시에만 설정 가능
2. 런타임 SM 변경은 불가 → 시간 크레딧으로 보상
3. MPS 필수 (SM 파티셔닝용)

### 향후 개선
1. CUDA 12.x의 Green Context 활용 (동적 SM 조정)
2. CUPTI 통합으로 실시간 SM 사용률 모니터링
3. ML 기반 예측 모델로 선제적 재분배

---

## 8. 구현 결과 (2024-12)

### 8.1 구현된 파일

```
controller/
├── sm_manager.py              # SM 할당 및 Work-conserving 관리
├── scheduling_coordinator.py  # SM + Time Credit 통합 코디네이터
├── profiler_client.py         # Profiler REST API 클라이언트
└── scheduler.py               # 기존 스케줄러 (libbless 통신)

tests/
├── test_colocation.py         # Co-location 테스트
└── test_integration.py        # 통합 테스트 (16개 테스트)
```

### 8.2 실제 테스트 결과

#### Co-location SM 할당 (84 SMs, RTX 4090)
| 조합 | Workload A | Workload B | 비율 |
|------|-----------|-----------|------|
| COMPUTE + MIXED | 46 SM (55%) | 37 SM (45%) | 55:45 |
| COMPUTE + COMPUTE | 42 SM (50%) | 42 SM (50%) | 50:50 |
| MIXED + MIXED | 42 SM (50%) | 42 SM (50%) | 50:50 |

#### 시간 보상 계수
- SM 55% 이상: 1.0x (보상 없음)
- SM 45%: 1.11x
- SM 40%: 1.25x
- SM 30%: 1.67x (최대 2.0x 제한)

#### Work-conserving 부스트
- 초기: 1.0x
- 파트너 idle: +0.55x (SM% 기반)
- 파트너 deallocate: 추가 +0.45x
- 최종: 최대 2.0x

### 8.3 사용법

```python
from scheduling_coordinator import SchedulingCoordinator

# 코디네이터 생성
coord = SchedulingCoordinator(total_sms=84)

# Co-location 설정
state_a, state_b = coord.setup_colocation(
    "tenant_A", "tenant_B",
    model_a="resnet50", model_b="bert"
)

# 크레딧 계산
credits = coord.compute_credits(["tenant_A", "tenant_B"], 524288)

# Work-conserving 트리거
coord.handle_workload_idle("tenant_A")
```
