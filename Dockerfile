FROM python:3.8
WORKDIR /usr/src/app/
COPY requirements.txt /usr/src/
RUN apt-get update
RUN pip install -r /usr/src/requirements.txt
RUN pip install torch==1.8.0 -f https://download.pytorch.org/whl/torch_stable.html
CMD [ "python", "api.py", "3080" ]
