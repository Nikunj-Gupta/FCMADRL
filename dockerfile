FROM ubuntu:16.04 

RUN apt-get update 
RUN apt-get install vim -y 
RUN apt-get install curl -y 
RUN apt install git -y 
RUN apt install python-pip -y && pip install --upgrade pip 

RUN cd home && git clone https://github.com/openai/gym && cd gym && pip install -e . 

#RUN pip install tensorflow && pip install keras && pip install statistics && pip install matplotlib 

COPY ./requirements.txt /home 
RUN cd /home && pip install -r requirements.txt 

RUN cd home && git clone https://github.com/openai/multiagent-particle-envs.git && cd multiagent-particle-envs && pip install -e . 

RUN cd home && git clone https://github.com/Nikunj-Gupta/FCMADRL.git && cd .. 