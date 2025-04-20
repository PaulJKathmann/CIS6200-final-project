import boto3
import json
from botocore.exceptions import ClientError

class BedrockAPIWrapper:
    def __init__(self, model_id="meta.llama3-70b-instruct-v1:0"):
        """
        Initialize the BedrockAPIWrapper with the specified model ID.
        """
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime")

    def generate_text(self, prompt, max_token_count=512, temperature=0.5, top_p=0.9):
        """
        Generate text using the Amazon Bedrock API.

        :param prompt: The input text prompt for the model.
        :param max_token_count: Maximum number of tokens to generate.
        :param temperature: Sampling temperature for text generation.
        :param top_p: Top-p sampling parameter.
        :return: Generated text from the model.
        """
        # Format the request payload using the model's native structure.
        native_request = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_token_count,
                "temperature": temperature,
                "topP": top_p,
            },
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)

        try:
            # Invoke the model with the request.
            response = self.client.invoke_model(modelId=self.model_id, body=request)

            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract and return the response text.
            return model_response["results"][0]["outputText"]

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            return None