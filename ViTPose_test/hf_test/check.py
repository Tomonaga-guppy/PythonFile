from huggingface_hub import HfApi

api = HfApi()
query = "vitpose wholebody 133"

results = api.list_models(search=query, limit=30)
for m in results:
    print(m.modelId)
