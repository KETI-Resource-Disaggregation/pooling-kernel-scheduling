Pooling Kernel Scheduling – Quick Start

다중 테넌트 커널 스쿼드 스케줄링을 빠르게 실행하는 방법만 정리했습니다. (상세 원리·분석은 생략)

0) 준비
# 리포 루트
cd pooling-kernel-scheduling

# CUDA 헤더가 보이는 환경에서 인터셉터 빌드
cd libbless && make && cd ..

1) 마스터(스케줄러) 실행
# 스쿼드/백로그/분배 설정 (원하는 값으로 조정)
export SQUAD=1000
export BACKLOG="A:1000,B:2000,C:1000"
export SQUAD_DIST="A:334,B:333,C:333"   # 균등 예시

# 이전 소켓/로그 정리(권장)
rm -f /tmp/bless-master.sock squad_log.csv

# 실행
PYTHONUNBUFFERED=1 python3 controller/scheduler.py


마스터 소켓: /tmp/bless-master.sock

로그: squad_log.csv (SQUAD_START/END, BOOST_ON/OFF 등)

2) 테넌트 실행 (A/B/C 각각 터미널 하나씩)

공통 환경:

cd pooling-kernel-scheduling

export LD_PRELOAD=$PWD/libbless/libbless.so
export BLESS_MASTER=/tmp/bless-master.sock
export BLESS_SAFE_GEMM=1       # 안전 매트멀(권장)


A:

export BLESS_TENANT=A
export BLESS_LIMIT_PCT=25

python3 controller/run_train.py --model gpt2 --steps 100 \
  --batch 8 --seq 384 --d_model 1024 --n_layer 12 --n_head 16 --ff_mult 4


B:

export BLESS_TENANT=B
export BLESS_LIMIT_PCT=50

python3 controller/run_train.py --model gpt2 --steps 100 \
  --batch 8 --seq 384 --d_model 1024 --n_layer 12 --n_head 16 --ff_mult 4


C:

export BLESS_TENANT=C
export BLESS_LIMIT_PCT=25

python3 controller/run_train.py --model gpt2 --steps 100 \
  --batch 8 --seq 384 --d_model 1024 --n_layer 12 --n_head 16 --ff_mult 4


VRAM 피크로 실패하면 --seq / --d_model / --n_layer / --batch 를 줄이세요.

3) 출력·로그

마스터: squad_log.csv

테넌트: A.csv, B.csv, C.csv (스텝·토큰/초 등)

4) 흔한 이슈 (아주 간단히)

마스터가 곧바로 all done → 테넌트에서 BLESS_MASTER/LD_PRELOAD 설정 확인, 소켓 삭제 후 재실행.

부스트가 안 보임 → SQUAD_DIST/BACKLOG 설정 및 테넌트 실행 상태 확인.

CUDA illegal access → 위 모델 파라미터 축소, BLESS_SAFE_GEMM=1 유지.


5) 시각화 방법
python viz_squad_window.py --focus-squad 880 --span 15

