FROM public.ecr.aws/lambda/python:3.12-arm64

COPY app.py ${LAMBDA_TASK_ROOT}/

RUN pip install boto3 --no-cache-dir

CMD ["app.handler"]
