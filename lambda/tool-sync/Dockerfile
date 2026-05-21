FROM public.ecr.aws/lambda/python:3.12

COPY index.py ${LAMBDA_TASK_ROOT}/

RUN pip install boto3 psycopg2-binary httpx --no-cache-dir

CMD ["index.handler"]
