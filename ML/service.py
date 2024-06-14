import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON

classifier = bentoml.sklearn.get(tag_like="classifier:latest").to_runner()

service = bentoml.Service("iris_classifier", runners=[classifier])


@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(features):
    return classifier.predict.run(features)


if __name__ == "__main__":
    bentoml.serve(service)
