FROM python:3.9-slim

WORKDIR /tester

COPY . .

RUN pip install numpy pandas scikit-learn

CMD ["python", "tester.py"]
