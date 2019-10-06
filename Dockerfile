FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y vim

RUN pip install keras
RUN pip install pillow pandas xgboost sklearn matplotlib

WORKDIR /NN/

CMD ["/bin/bash"]
#CMD [ "train.py" ]