FROM python:3.9-slim

WORKDIR /trainer

COPY . .

RUN pip install numpy pandas scikit-learn

CMD ["python", "trainer.py"]
