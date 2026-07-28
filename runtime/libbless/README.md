# libbless — prism/runtime 내재화 본 (시분할 토큰버킷 인터셉터)

## 출처 (정본)
- **KETI-Resource-Disaggregation / `pooling-kernel-scheduling`** 의 libbless 정본
  (원본 경로: `archive/pooling-kernel-scheduling/libbless/libbless.cpp`, 1219줄).
- 본 폴더는 그 **검증 끝난 정본을 prism/runtime 안으로 복사(self-contained)** 한 것.
  원본 archive/ 는 그대로 유지(이동 아님).

## 적용된 수정 (Exp_7 검증 → Exp_8 정본 반영)
1. **line 308 stray `']'` 오타 수정** (`}]`→`}`). 이 오타 때문에 정본 소스가 빌드 불가였음.
2. **Fix1 — credit 게이트 언더플로 클램프** (`kernel_credit_gate`).
   기존 `fetch_sub(BURST)`+undo는 잔량<BURST일 때 `credit_remain`을 음수로 만들고
   그 음수가 다음 게이트에서 `cr<0`="무제한"으로 오인되어 시분할 비례가 깨졌다.
   → CAS로 정확히 `min(cur,BURST)`만 차감(0 미만 금지)하도록 수정.
   효과: GPU 런치 천장 이내(in-envelope)에서 weight 비례 시분할이 정확·결정적
   (2:1→0.667, 3:1→0.750). 상세: `prism/reports/Exp_7_credit_gate_fairness_230/`.

## 적용된 수정 2 (Exp_16 검증 → Exp_18 정본 반영, 2026-07-04)
3. **time_credit 보강 3건** (`time_credit_gate`/`time_batch_end_and_charge`/소켓 핸들러):
   - **무제한 플래그 분리** (`time_unlimited` atomic) — 원본은 `credit<0`=무제한이라
     차감으로 적자가 되는 순간 게이트·차감이 모두 우회(Exp_16 실측: 가중 전부 무효과).
     적자는 "빚"으로 게이트를 계속 막고, 무제한은 플래그로만 표현.
   - **차감 무조건화** — 무제한이 아니면 적자로도 차감(빚 이월). 원본 `credit≥0` 조건 제거.
   - **대기 전 배치 청구** — 게이트 대기 직전 열린 배치를 청구·마감. 대기 시간이
     사용으로 청구되던 결함(청구≈벽시계 전체) 차단.
   효과: 이종 페어(prefill×decode) 시간 비율 제어 성립 — 목표 점유 0.333~0.750 에
   오차 ≤0.007, 시간×SM(MPS) 직교, 3-tenant 3-way 확장(오차 ≤0.006).
   상세: `prism/reports/Exp_16_time_credit_230/`(검증) · `Exp_17_generalization_230/`(일반화)
   · `Exp_18_canonical_time_reflect_230/`(정본 반영·재현·회귀).

## 미반영 상태 (★ 범위 밖, 추후 별도 결정)
- **Fix2 (청크 입도 `cur>>2`)** — 미반영.
- **Fix3 (`credit_refill` 이월+cap 명령 + 스케줄러 전환)** — 미반영.
  → 본 내재화 본은 **오타 + Fix1 + time_credit 보강 3건** (= Exp_18 정본 반영본).

## 파일
| 파일 | md5 | 비고 |
|---|---|---|
| `libbless.cpp` | `ee7e63e76c6d42075980fbede7bb72eb` | 정본 소스(오타+Fix1+time 보강 3건) |
| `libbless.so` | `715023af3dbbd146d3646cba85bb3f8d` | 소스에서 재빌드(60424 B) — Exp_16 검증 빌드와 byte-identical |
| `libbless.cpp.bak` / `libbless.so.bak` | `daa4a6fa…` / `c7cfbce1…` | Exp_18 반영 전 백업(롤백용, Exp_8 반영본) |
| `Makefile` | — | 빌드 (Exp_6/7/8 동일 플래그) |
| `context_manager.hpp`, `routing.hpp` | — | Makefile 전제 파일(현 libbless.cpp는 직접 포함 안 함, 호환 위해 동봉) |

구 계보(Exp_8~Exp_17 문서·메모리의 "정본 md5 `daa4a6fa`/`c7cfbce1`")는 Exp_18 반영 시점까지의
값 — 현재 정본은 위 표(`ee7e63e7`/`715023af`)가 기준. `archive/pooling-kernel-scheduling/libbless/`
사본은 Exp_8 시점(`daa4a6fa`/`c7cfbce1`) 그대로 무변경(최신 정본은 본 폴더).

## 빌드 (소스에서 재생성 가능 — 정본-바이너리 불일치 방지)
```bash
cd prism/runtime/libbless
make            # = g++ -fPIC -O2 -std=c++17 -I/usr/local/cuda/include \
                #       -o libbless.so libbless.cpp -shared -L/usr/local/cuda/lib64 \
                #       -lcuda -lcudart -ldl -lpthread
# 결과: libbless.so 60424 B (md5 715023af). rc=0.
```
요구: CUDA 12.x (`/usr/local/cuda`), g++ C++17. 동봉 `libbless.so`는 위 명령으로 언제든 재생성 가능.

## 사용 (LD_PRELOAD 인터셉션)
```bash
LD_PRELOAD=.../prism/runtime/libbless/libbless.so BLESS_TENANT=A python3 <workload>
# 제어 소켓: /tmp/bless-{pid}.sock  (credit_set / mem_quota / set_route / time_* 등)
# SM 비율(공간분할)은 MPS 데몬 하에서 BLESS_LIMIT_PCT로 실효 — Exp_9 참조.
```

## 관련 실험
- Exp_6: libbless 4기능 베이스라인. Exp_7: credit 게이트 3수정·envelope.
- Exp_8: 정본에 오타+Fix1 반영·재빌드. Exp_9: MPS 하 SM 비율 실효.
- Exp_11(본 내재화): `prism/reports/Exp_11_libbless_internalize_230/`.
