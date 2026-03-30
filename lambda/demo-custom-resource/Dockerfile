FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY index.py demo_invoice_schema.json ${LAMBDA_TASK_ROOT}/

CMD ["index.handler"]
