FROM python:3.9-slim
ENV PYTHONPATH=/usr/lib/python3.9/site-packages

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils libgl1 libglib2.0-0 \
    python3-pip \
    && apt-get install -y gcc musl-dev python3-dev \
    && apt-get install psmisc \
    && apt-get clean \
    && apt-get autoremove

RUN pip3 install --upgrade pip
RUN pip3 install wandb
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY run.py ./
COPY image.png ./
CMD ["python", "run.py"]
