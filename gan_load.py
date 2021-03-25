# Test code for checking 

import torch
import base64
import io
from PIL import Image
import torchvision.transforms as transforms
from model import Generator

from torchvision.utils import save_image

import boto3
import json

# settings
weight_path = "/home/ubuntu/murata/Server/ganserve/serve/cyclegan_generator_a2b.pth"
image_path = "/home/ubuntu/murata/Server/ganserve/serve/examples/image_classifier/kitten.jpg"

# define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# model settingds
model = Generator()
model.load_state_dict(torch.load(weight_path))
model.to(device)
model.eval()

# create send data
data = open(image_path, "rb")
body = data.read()

## preprocess()
image_processing = transforms.Compose([
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
image = body


if isinstance(image, str):
    print("bytestring")
    image = base64.b64decode(image)
            
if isinstance(image, (bytearray, bytes)):
    print("bytearray")
    image = Image.open(io.BytesIO(image))
    image = image_processing(image)
else:
    print("numpy array")
    image = torch.FloatTensor(image)


images = [image]
preprocess_output = torch.stack(images).to(device)
# print(preprocess_output.shape)


## inference()
infer_output = model(preprocess_output)
# print(infer_output)


# save image
save_path = "/home/ubuntu/murata/Server/ganserve/serve/image_dir/target.png" 
save_image(infer_output, save_path, nrow=3, normalize=True)


# aws settings
bucket_location = "us-east-1"
bucket = "murata-torchserve-db"
object_name = "target.png"
file_name = object_name

s3_client = boto3.resource("s3")

## postprocess()
output_bytearray = open(save_path, "rb").read()
url = "https://s3-{}.amazonaws.com/{}/{}".format(bucket_location, bucket, object_name)

output_dict = {}

print("ok")
s3_client.Bucket(bucket).upload_file(Filename=save_path, Key=object_name)

output_dict["url"] = url
# output_dict["bytearray"] = output_bytearray
print(output_dict)

output_json = json.dumps(output_dict)
print(output_json)
