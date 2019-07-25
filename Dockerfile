# docker build -t ubuntu1604py36
FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
RUN python3.6 -m pip install wheel

# Install context bandit task runner
RUN mkdir -p /home/contextual-bandit-recommender
RUN git clone https://github.com/dhfromkorea/contextual-bandit-recommender.git
RUN cd /home/contextual-bandit-recommender
RUN pip install -r requirements.txt

WORKDIR /home/contextual-bandit-recommender
CMD ["python", "main.py" ]

