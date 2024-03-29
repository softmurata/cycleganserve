import torch
from torchvision.utils import save_image
from torchvision import transforms
import os
import boto3
import json
import base64
import io
from PIL import Image

from model import Generator

from ts.torch_handler.base_handler import BaseHandler

"""
Overwrite BaseHandler class

cannot read model well

"""

class ModelHandler(BaseHandler):
    """[summary]

    Args:
        BaseHandler ([type]): [description]

    Returns:
        [type]: [description]
    """

    image_processing = transforms.Compose([
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0

        self.image_filename = "target.png"
        self.bucket_location = "us-east-1"
        self.bucket = "murata-torchserve-db"


    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
	    First try to load torchscript else load eager mode state_dict based model.
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """

        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )
        self.manifest = context.manifest

        # load model
        # ToDo: you have to change absolute path for weight file
        model_pt_path = "/Users/tatsuro/Documents/Media2CloudTutorial/Server/cycleganserve/serve/examples/cycleganserve/cyclegan_generator_a2b.pth"  # generator weight
        # model_pt_path = "/home/ubuntu/murata/Media2Cloud/Server/cycleganserve/serve/examples/cycleganserve/cyclegan_generator_a2b.pth"
        self.model = Generator()
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.map_location))
        self.model.to(self.device)
        self.model.eval()

        
        self.initialized = True

    def preprocess(self, data):
        """[summary]
        receive binary data and covert it into image format
        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """

        print("preprocess")
        images = []

        for row in data:
            image = row.get("data") or row.get("body")

            if isinstance(image, str):
                image = base64.b64decode(image)
            
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                image = torch.FloatTensor(image)

            images.append(image)
        
        return torch.stack(images).to(self.device)

    def inference(self, model_input):
        """[summary]
        model inferences against input
        Args:
            model_input ([type]): [description]

        Returns:
            [type]: [description]
        """
        print("inference")
        model_output = self.model(model_input)
        return model_output
    
    def postprocess(self, inference_output):
        """[summary]
        output for rest api
        basically json
        Args:
            inference_output ([type]): [description]

        Returns:
            [type]: [description]
        """
        postprocess_output = inference_output
        # convert results into json format

        # ToDo: change absolute path for image dir
        file_name = "/Users/tatsuro/Documents/Media2CloudTutorial/Server/cycleganserve/serve/image_dir/{}".format(self.image_filename)
        # file_name = "/home/ubuntu/murata/Media2Cloud/Server/cycleganserve/serve/image_dir/{}".format(self.image_filename)
        save_image(postprocess_output, file_name)

        
        # s3 upload
        json_format = self.upload_file(file_name, self.bucket, object_name=None)
        # url = "https://{}.s3.amazonaws.com/{}".format(self.bucket, self.image_filename)
        # json_format = {"result": 0.99, "url": url}

        return json_format

    def upload_file(self, file_name, bucket, object_name):
        if object_name is None:
            object_name = file_name

        s3_client = boto3.resource("s3")

        url = "https://{}.s3.amazonaws.com/{}".format(self.bucket, self.image_filename)

        s3_client.Bucket(self.bucket).upload_file(Filename=file_name, Key=self.image_filename)
        
        return json.dumps({"url": url})


# main function
_service = ModelHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return [data]
    except Exception as e:
        raise e







