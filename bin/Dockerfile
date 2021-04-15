FROM patkraw/mask-detecton-wf:latest

RUN mkdir crisis
WORKDIR /crisis

RUN pip install opencv-python
RUN pip install numpy
RUN pip install pandas
RUN pip install nltk
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install Pillow
RUN pip install optuna==2.0.0
RUN pip install tensorflow
RUN pip install keras

RUN mkdir ./bin

COPY add_labels.py ./bin
RUN chmod 777 ./bin/add_labels.py
COPY preprocess_images.py ./bin
RUN chmod 777 ./bin/preprocess_images.py
COPY preprocess_tweets.py ./bin
RUN chmod 777 ./bin/preprocess_tweets.py
COPY split_tweets.py ./bin
RUN chmod 777 ./bin/split_tweets.py
COPY train_resnet.py ./bin
RUN chmod 777 ./bin/train_resnet.py
COPY hpo_train_resnet.py ./bin
RUN chmod 777 ./bin/hpo_train_resnet.py
COPY utils.py ./bin
RUN chmod 777 ./bin/utils.py