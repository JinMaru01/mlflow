import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

print(os.getenv("HOST"))
print(os.getenv("TRACKING_URI"))
print(os.getenv("MLFLOW_S3_ENDPOINT_URL"))