# Python 람다 함수를 위한 Dockerfile
FROM python:3.13.4
# Poetry 설치
RUN pip install -U poetry
# 경로 정의
WORKDIR /workdir
# 로컬에 있는 pyproject.toml, poetry.lock 파일을 컨테이너로 복사
COPY poetry.lock pyproject.toml /workdir/
# Poetry를 이용하여 의존성 설치
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction
# 로컬에 있는 소스코드를 컨테이너로 복사
COPY . /workdir
# Python 경로 설정
# ENV PYTHONPATH=/usr/local/bin/python3.13

# 환경 변수 설정 (RDS 정보)
# ENV DATABASE_HOST=.rds.amazonaws.com
ENV MYSQL_MAIN_USERNAME=${MYSQL_MAIN_USERNAME}
ENV MYSQL_MAIN_PASSWORD=${MYSQL_MAIN_PASSWORD}
# ENV DATABASE_DB=
# ENV DATABASE_PORT=33067

# S3 설정
ENV CREDENTIALS_ACCESS_KEY=${CREDENTIALS_ACCESS_KEY}
ENV CREDENTIALS_SECRET_KEY=${CREDENTIALS_SECRET_KEY}
ENV AWS_REGION=${AWS_REGION}
ENV S3_BUCKET_NAME=${S3_BUCKET_NAME}

# Poetry 바이너리 권한 확인 및 설정
# RUN chmod +x /usr/local/bin/poetry
# Poetry가 설치된 Python을 사용하도록 설정
# RUN sed -i '1s|^.*$|#!/usr/local/bin/python3.13|' /usr/local/bin/poetry
# 권한과 바이너리 위치 확인
# RUN ls -l /usr/local/bin/poetry
# WORKDIR /workdir
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]