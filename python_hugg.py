

import requests

# Replace 'YOUR_API_TOKEN' with your actual API token
API_TOKEN = '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'

# URL of your deployed model on Hugging Face Spaces
MODEL_NAME = 'VASUGI/AGOR_BRAIN_AI'

# API endpoint for model inference
API_ENDPOINT = f'https://huggingface.co/api/inference/VASUGI/AGOR_BRAIN_AI'

# Sample input data
input_data = {
    'inputs': 'Sample input text for inference.'
}

# Send POST request to the Hugging Face API endpoint
response = requests.post(
    API_ENDPOINT,
    headers={'Authorization': f'Bearer {API_TOKEN}'},
    json=input_data
)

# Parse and print the response
if response.status_code == 200:
    predictions = response.json()
    print(predictions)
else:
    print(f"Failed to make request: {response.status_code} - {response.text}")
