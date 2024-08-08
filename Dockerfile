
FROM public.ecr.aws/lambda/python:3.12

RUN microdnf update -y && microdnf install -y gcc-c++ make


COPY requirements.txt ${LAMBDA_TASK_ROOT}


RUN pip install -r requirements.txt


COPY travelAgent.py ${LAMBDA_TASK_ROOT}


RUN chmod +x travelAgent.py


CMD ["travelAgent.lambda_handler","streamlit", "run", "travelAgent.py"]

