# libbless 빌드 환경 노트 (Exp_65)

**정본은 소스다**: `libbless.cpp` md5 `2b4d721f` (Exp_61 정본 계보:
`077dae7d`(Exp_51) → `2b4d721f`(Exp_61)).

이 디렉토리의 `libbless.so` (md5 `311360d1`)는 **구 230 환경 빌드본**이다:

| 항목 | 빌드 당시 (230, ~Exp_61) | 현 232 | 현 233 |
|---|---|---|---|
| GPU | RTX A6000 ×2 | RTX PRO 6000 Blackwell | RTX 5090 |
| driver / CUDA | 구 환경 (CUDA 12.x) | 590.48.01 / 13.1 (툴킷 12.9·13.1) | 580.173.02 / 13.0 (**툴킷 없음**) |
| glibc | 2.35 | 2.35 | 2.39 |

→ **232·233 어디서도 이 .so를 그대로 쓰지 말 것.** 배포 전 대상 노드 환경으로
재빌드하고, 새 md5를 계보에 등재한다(선례: Exp_52의 232판 `cf69c8fb` — 현재 소재 불명).

**[Exp_70 갱신]** 트리의 `.so`는 232 재빌드본으로 교체됨:
- `libbless.so` md5 **`92fdb6ba`** — cpp `2b4d721f` + CUDA 12.9(V12.9.41) + glibc 2.35
  + driver 590.48.01, dlsym cuCtxCreate_v4/v3 분기(O-4) 포함. `make CUDA_HOME=/usr/local/cuda-12.9`
- `runtime/gating_lib.so` md5 **`21a17551`** — Exp_42 기록의 232 빌드 md5와 일치(재현 빌드 검증)
- 구 230판 `311360d1`은 git 이력( ~Exp_69)과 tar 백업에 보존.

**[Exp_73 갱신]** `libbless.so` md5 **`ffb78c13`** — Exp_72 판(92fdb6ba)에 `BLESS_STATS_LOG`
time_stats 파일 발화 추가(feeder occupancy 배선). CUDA 12.9 빌드 동일. 이미지 exp73.

.so를 계속 커밋하는 이유: 소용량 + md5 계보가 보고서 전반에서 참조되는 추적
대상(Exp_46 판단, `.gitignore` 예외 규칙 참조).
