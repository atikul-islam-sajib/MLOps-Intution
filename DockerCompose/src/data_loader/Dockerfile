FROM python:3.9-slim

WORKDIR /data_loader

COPY . .

RUN pip install numpy pandas scikit-learn

CMD ["python", "load_data.py"]
