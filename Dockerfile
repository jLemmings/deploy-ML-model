FROM python:3.6-slim
COPY ./main.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./data/model.pickle /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]