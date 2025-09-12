# ---------- 1단계: 빌드 이미지 ----------
FROM python:3.10.11-slim AS builder

# 작업 디렉토리 설정
WORKDIR /build

# Poetry 설치
RUN pip install --no-cache-dir poetry

# 의존성 파일 복사
COPY poetry.lock pyproject.toml ./

# Poetry 설정 및 의존성 설치
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction --no-cache

# 앱 소스 코드 복사
COPY . .

# ---------- 2단계: 실행 이미지 ----------
FROM python:3.10.11-slim AS runtime

# 작업 디렉토리 설정
WORKDIR /app

# 빌드 이미지에서 설치된 라이브러리와 앱 코드 복사
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

# 애플리케이션 실행
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]