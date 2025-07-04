FROM python:3.10-slim

ENV MLFLOW_HOME=/mlflow
ENV BACKEND_STORE_URI=postgresql://mlflow:mlflow@mlflow-db:5432/mlflow
ENV ARTIFACT_ROOT=s3://mlflow-artifacts/

WORKDIR ${MLFLOW_HOME}

RUN mkdir -p /mlflow/artifacts

RUN pip install --no-cache-dir mlflow[extras]==2.13.0 psycopg2-binary boto3

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri=${BACKEND_STORE_URI} \
    --default-artifact-root=${ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts
