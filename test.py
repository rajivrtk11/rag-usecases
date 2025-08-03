from vertexai.preview.generative_models import list_models
import vertexai

vertexai.init(project="your-gcp-project-id", location="us-central1")

models = list_models()

for model in models:
    print(model.name)
