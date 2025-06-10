workers = 3

# 워커 클래스는 Uvicorn
worker_class = "uvicorn.workers.UvicornWorker"

max_requests = 10000
max_requests_jitter = 100

# 타임아웃 제거 (무제한)
timeout = 0

# 워커 임시 파일 RAM 파일 시스템에 저장 (RAM 이용량 모니터링 필요!) Linux용 옵션
# worker_tmp_dir = "/dev/shm"

keepalive = 10


# preload로 COW 기반 메모리 최적화
preload_app = True

loglevel = "info"
bind = "0.0.0.0:8001"
