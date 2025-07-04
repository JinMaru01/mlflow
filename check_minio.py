import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://10.120.210.56:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin"
)

response = s3.list_objects_v2(Bucket="mlflow-artifacts")
for obj in response.get("Contents", []):
    print(obj["Key"])
