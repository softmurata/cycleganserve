# Media2Cloud
# EndPoint
https://8rsjf9kpd9.execute-api.us-east-1.amazonaws.com/demo

# s3 ingest bucket(media2cloud)
so0050-media2cloud-cc591708db16-ingest

# s3 bucket(torchserve)
region_name: us-east-1
bucket_name:murata-torchserve-db

# uuid
1. 9848945a-5448-ead8-2d1b-0e704c26e800
2. 6f1a1abb-1dd8-1230-400f-b3fb42dff91f



# torchserve
# settings
1. create model name directory under Server
```
cd Server
mkdir <model-name>
cd <model-name>
```
2. clone repository
```
git clone https://github.com/pytorch/serve.git
```
3. execute install scripts
```
cd serve
python ./ts_scripts/install_dependencies.py --cuda=cu101
pip install torchserve torch-model-archiver
```

# multiple model
In torchserve directory, you can host only one model. So if you want to multiple model in one server, you should cut into multiple directory.

examples is following
```
Server - alexnet_server - serve - model_store
                                - ts
                                - examples
                                - ...
       - densenet161_server - serve - model_store
                                    - ts
                                    - examples
                                    - ...
       - cyclegan_server - serve - model_store
                                 - ts
                                 - examples
                                 - ...
```

You can switch this model by following command. But You have to prepare .mar file in model_store directory in advance.

ex) alexnet
```
cd Server/alexnet_server/serve
torchserve --start --ncs --model-store model_store --models alexnet=alexnet.mar

```

if you want to stop
```
torchserve --stop
```

ex) densenet161
```
cd Server/densenet161_server/serve
torchserve --start --ncs --model-store model_store --models densenet161=densenet161.mar
```


## Image Classifier torcharchiver command
# DenseNet
1. get weight model file
```
cd serve
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
```
2. create .mar file
```
torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
```

3. create model_store directory and move .mar file under it
```
mkdir model_store
mv densenet161.mar model_store/
```

4. start command
```
torchserve --start --ncs --model-store model_store --models densenet161=densenet161.mar
```

5. stop command
```
torchserve --stop
```

# Alexnet
1. get weight model file
```
cd serve
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
```
2. create .mar file
```
torch-model-archiver --model-name alexnet --version 1.0 --model-file examples/image_classifier/alexnet/model.py --serialized-file alexnet-owt-4df8aa71.pth --export-path model_store --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
```

3. create model_store directory and move .mar file under it
```
mkdir model_store
mv alexnet.mar model_store/
```

4. start command
```
torchserve --start --ncs --model-store model_store --models alexnet=alexnet.mar
```

5. stop command
```
torchserve --stop
```

# ResNet18
1. get weight model file
```
cd serve
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
```
2. create .mar file
```
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file examples/image_classifier/resnet-18/model.py --serialized-file resnet18-5c106cde.pth --export-path model_store --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
```

3. create model_store directory and move .mar file under it
```
mkdir model_store
mv resnet-18.mar model_store/
```

4. start command
```
torchserve --start --ncs --model-store model_store --models resnet-18=resnet-18.mar
```

5. stop command
```
torchserve --stop
```

## Image Segmenter torcharchiver command

# FCN resnet101 coco
1. get weight model file
```
cd serve
wget https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth
```
2. create .mar file(with utils file version)
```
torch-model-archiver --model-name fcn_resnet_101 --version 1.0 --model-file examples/image_segmenter/fcn/model.py --serialized-file fcn_resnet101_coco-7ecb50ca.pth --handler image_segmenter --extra-files examples/image_segmenter/fcn/fcn.py,examples/image_segmenter/fcn/intermediate_layer_getter.py
```

3. create model_store directory and move .mar file under it
```
mkdir model_store
mv fcn_resnet_101.mar model_store/
```

4. start command
```
torchserve --start --ncs --model-store model_store --models fcn=fcn_resnet_101.mar
```

5. stop command
```
torchserve --stop
```

6. check command
```
curl -X POST http://localhost:8080/predictions/fcn -T examples/image_segmenter/fcn/persons.jpg
```
 => output (array, shape=(batch, height, width, 2), (class and probability))


## Object detector torcharchiver command
# Mask RCNN
1. get weight model file
```
cd serve
wget https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
```
2. create .mar file(with utils file version)
```
torch-model-archiver --model-name maskrcnn --version 1.0 --model-file examples/object_detector/maskrcnn/model.py --serialized-file maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --handler object_detector --extra-files examples/object_detector/index_to_name.json
```

3. create model_store directory and move .mar file under it
```
mkdir model_store
mv maskrcnn.mar model_store/
```

4. start command
```
torchserve --start --ncs --model-store model_store --models maskrcnn=maskrcnn.mar
```

5. stop command
```
torchserve --stop
```

6. check command
```
curl -X POST http://localhost:8080/predictions/maskrcnn -T examples/object_detector/persons.jpg
```
 => output
 ```
 [
  {
    "person": [
      169.61875915527344,
      50.14577865600586,
      300.843994140625,
      442.492919921875
    ],
    "score": 0.9995498061180115
  },
  {
    "person": [
      90.41835021972656,
      66.83667755126953,
      194.2113494873047,
      437.27752685546875
    ],
    "score": 0.9994066953659058
  },
  {
    "person": [
      362.3892517089844,
      158.00897216796875,
      521.0659790039062,
      385.55084228515625
    ],
    "score": 0.9953246116638184
  },
  {
    "handbag": [
      68.58448791503906,
      279.283935546875,
      111.14231872558594,
      400.91204833984375
    ],
    "score": 0.9939875602722168
  },
  {
    "person": [
      473.85400390625,
      147.2745819091797,
      638.3865356445312,
      364.52313232421875
    ],
    "score": 0.9903406500816345
  },
  {
    "handbag": [
      225.60440063476562,
      142.7439422607422,
      302.4504089355469,
      230.29791259765625
    ],
    "score": 0.9889532327651978
  },
  {
    "handbag": [
      380.28204345703125,
      259.18206787109375,
      419.51519775390625,
      318.27215576171875
    ],
    "score": 0.9680995941162109
  },
  {
    "bench": [
      273.4175109863281,
      217.40707397460938,
      441.253173828125,
      396.3609313964844
    ],
    "score": 0.9635116457939148
  },
  {
    "person": [
      541.36474609375,
      156.64715576171875,
      620.078857421875,
      249.49537658691406
    ],
    "score": 0.8153693675994873
  },
  {
    "chair": [
      455.21783447265625,
      207.56234741210938,
      491.1147155761719,
      274.7507629394531
    ],
    "score": 0.7720490097999573
  },
  {
    "person": [
      626.2461547851562,
      178.66172790527344,
      640.0,
      246.1094512939453
    ],
    "score": 0.714340090751648
  },
  {
    "dog": [
      557.7418212890625,
      202.899169921875,
      611.4240112304688,
      256.95574951171875
    ],
    "score": 0.7024562954902649
  },
  {
    "person": [
      359.2744140625,
      161.6046142578125,
      493.7586669921875,
      296.9685363769531
    ],
    "score": 0.7023776173591614
  },
  {
    "person": [
      548.9495849609375,
      177.09860229492188,
      640.0,
      364.51434326171875
    ],
    "score": 0.69603431224823
  },
  {
    "bench": [
      297.37701416015625,
      208.04835510253906,
      563.4820556640625,
      380.4136047363281
    ],
    "score": 0.6695427298545837
  },
  {
    "handbag": [
      412.6864929199219,
      272.4156494140625,
      459.142822265625,
      363.9853820800781
    ],
    "score": 0.6374452710151672
  },
  {
    "bench": [
      444.64892578125,
      204.42015075683594,
      627.00634765625,
      359.8998107910156
    ],
    "score": 0.6021174192428589
  }
]
 ```

## Text Classification torcharchiver method
1. Train model
```
cd serve/examples/text_classification
python run_script.py
```

In the python script file, download data(.csv), train model and save weight file(model.pt + source_vocab.pt)


2. create .mar file(with utils file version)
```
torch-model-archiver --model-name my_text_classifier --version 1.0 --model-file model.py --serialized-file model.pt --handler text_classifier --extra-files "index_to_name.json,source_vocab.pt"
```

3. create model_store directory and move .mar file under it
```
mkdir model_store
mv my_text_classifier.mar model_store/
```

4. start command
```
torchserve --start --ncs --model-store model_store --models my_tc=my_text_classifier.mar
```

5. stop command
```
torchserve --stop
```

6. check command
```
curl -X POST http://localhost:8080/predictions/my_tc -T examples/text_classifier/sample.txt
```
 => output
 ```
 {
  "World": 0.00017590356583241373,
  "Sports": 2.1884987290832214e-05,
  "Business": 0.9950227737426758,
  "Sci/Tec": 0.004779411945492029
}
 ```

## Text to speech synthesizer torcharchiver method
# Waveglow
1. create mar file(shell script)
```
cd Server/textspeechsynthesizeserve/serve/examples/text_to_speech_synthesizer

bash create_mar.sh
```

2. create model_store directory and move .mar file into it
```
mv model_store
mv waveglow_synthesizer.mar model_store/
```

3. start torchsever
```
torchserve --start --model-store model_store --models waveglow_synthesizer.mar
```

4. check command
In this model, if you get access to the url with text file, you can get audio file(.wav)
```
curl -X POST http://localhost:8080/predictions/waveglow_synthesizer -T sample_text.txt -o audio.wav
```

## Transformer torcharchiver method
# NMT Transformer

currently, the error occurs that pip cannot install the content of requirements.txt, but I have no idea to solve it. 


## original model(Cyclegan)
1. prepare model(python model file and .pt or .pth weight file)
   1.1 create model file(don't need utils file for model, all components of model have to be implemented in model.py)
   1.2 If you don't have pretrained weight file, you will have to train model and get weight file(You should assign weight file's format as .pt or .pth)

   1.3 create gan_handler.py(custom handler class file) and gan_load.py(test code for checking workflow)

   <Tips for implementation>
   In case of implementing custom handler, basically you should succeed the BaseHandler class.
   Training and Test code is so important that you refer to model.py and dataset.py in case of implementing handler.py
   hanlder.py is composed of one handler class and one handle function
   it has mainly 4 sub function
   0. __init__() => you should define image_processing variables in this function
   1. initialize()
      explanation: construct model
   2. preprocess()
      explanation: preprocess image(refering to database.py)
   3. inference()
      explanation: AI model inferences(refering to train.py or test.py)
   4. postprocess()
      explanation: define output format as REST API

   ex) gan_handler.py
   ```
    import torch
    from torchvision.utils import save_image
    from torchvision import transforms
    import os
    # for uploading file
    import boto3
    import json
    import base64
    import io
    from PIL import Image

    from model import Generator

    from ts.torch_handler.base_handler import BaseHandler

    """
    Overwrite BaseHandler class

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
            model_pt_path = "/home/ubuntu/murata/Server/ganserve/serve/cyclegan_generator_a2b.pth"  # generator weight
            self.model = Generator()
            self.model.load_state_dict(torch.load(model_pt_path, map_location=self.map_location))
            self.model.to(self.device)
            self.model.eval()

            
            self.initialized = True

        def preprocess(self, data):
            """[summary]
            receive binary data and covert it into image format(This implementation is common if you deal with image as input at REST API)
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
            print("inference")
            model_output = self.model(model_input)
            return model_output
        
        def postprocess(self, inference_output):
            postprocess_output = inference_output
            # convert results into json format
            file_name = "/home/ubuntu/murata/Server/ganserve/serve/image_dir/{}".format(self.image_filename)
            save_image(postprocess_output, file_name)

            # s3 upload
            # json_format = self.upload_file(file_name, self.bucket, object_name=None)

            return json_format

        # sub method(You can create customize sub function in handler class)
        def upload_file(self, file_name, bucket, object_name):
            if object_name is None:
                object_name = file_name

            s3_client = boto3.resource("s3")

            url = "https://s3-{}.amazonaws.com/{}/{}".format(self.bucket_location, self.bucket, object_name)

            s3_client.Bucket(self.bucket).upload_file(Filename=file_name, Key=self.image_filename)
            
            return json.dumps({"url": url})


    # Main function
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


   ```

   ex) gan_load.py
   ```
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
    url = "https://{}.s3.amazonaws.com/{}".format(bucket, self.image_filename)

    output_dict = {}

    print("ok")
    s3_client.Bucket(bucket).upload_file(Filename=save_path, Key=object_name)

    output_dict["url"] = url
    # output_dict["bytearray"] = output_bytearray
    print(output_dict)

    output_json = json.dumps(output_dict)
    print(output_json)

   ```

2. clone serve repository and install dependencies
   2.0 create server repository
   ```
   mkdir ganserve
   cd ganserve
   ```
   2.1 clone serve repository
   ```
   git clone https://github.com/pytorch/serve.git
   ```
   2.2 create python virtual env and activate(In my environment, python3.7.6)
   ```
   python -m venv venv
   source venv/bin/activate
   ```
   2.3 install dependencies
   ex) examples of cuda10.1
   ```
   cd serve
   python ./ts_script/install_dependencies.py --cuda=cu101
   ```
   2.4 install torch-model-archiver and torchserve
   ```
   pip install torchserve torch-model-archiver
   ```
3. download or move weight file and model file under serve directory
   3.0 download weight file(under serve directory, put on .pt or .pth file)
   3.1 create gan diretory under examples
   ```
   cd examples
   mkdir gan
   cd gan
   mkdir cyclegan
   ```
   3.2 move required file under cyclegan
   ```
   model.py(Generator class and ResBlock class)
   gan_handler.py(custom handler class)
   gan_load.py(test code for loading model and confirming results)
   ```
4. create mar file by torch-model-archiver
```
torch-model-archiver --model-name cyclegan --version 1.0 --model-file examples/gan/cyclegan/model.py --serialized-file cyclegan_generator_a2b.pth --handler examples/gan/cyclegan/gan_handler.py
```
5. You have to create model_store directory and move into .mar file under model_store directory
```
mkdir model_store
mv cyclegan.mar model_store/
```

6. You should start torchserve server
ex) localhost
  torchserve start command
  ```
   torchserve --start --ncs --model-store model_store --models cyclegan=cyclegan.mar
  ```

  check command
  caution) you should prepare image data and go to the directory which includes it.
  ```
  curl -X POST http://localhost:8080/predictions/cyclegan -T kitten.jpg
  ```

ex) public access
    you should create config.properties under serve directory
    ```
    touch config.properties
    vim config.properties
    inference_address=http://0.0.0.0:8080
    ```

    torchserve start command
    ```
    torchserve --start --ncs --model-store model_store --models cyclegan=cyclegan.mar --ts-config config.properties
    ```

    check command
    ```
    curl -X POST http://{global_ip_adress}:8080/predictions/cyclegan -T kitten.jpg
    ```
    ex)
    AWS EC2 g4dn.xlarge instance wuth ipadress 50.17.23.119
    ```
    curl -X POST http://50.17.23.119:8080/predictions/cyclegan -T kitten.jpg
    ```


# python version REST API
if you use this rest api in another python project, following python code is useful.

ex) restapi.py
```
# requests with upload image file
import argparse
import urllib.request
import json

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default="kitten_small.jpg")
parser.add_argument("--ip_address", type=str, default="3.236.97.79")
parser.add_argument("--port", type=str, default="8080")
parser.add_argument("--model_name", type=str, default="alexnet", help="1. alexnet, 2. densenet161")
args = parser.parse_args()

if args.image_path.split(".")[-1] == "jpg":
    ext = "jpg"
elif args.image_path.split(".")[-1] == "png":
    ext = "png"
else:
    ext = "jpeg"
    
url = "http://{}:{}/predictions/{}".format(args.ip_address, args.port, args.model_name)
data = open(args.image_path, "rb")
reqbody = data.read()
data.close()
print(data)
files = {"image_file": data}

req = urllib.request.Request(
    url,
    reqbody,
    method="POST",
    headers={"Content-Type": "application/octet-stream"}   
)

with urllib.request.urlopen(req) as res:
    print(json.loads(res.read()))


```

run this command
ex) alexnet
```
cd Server/alexnet_server/serve
torchserve --start --ncs --model-store model_store --models alexnet=alexnet.mar
cd <python rest api file>
python restapi.py --ip_adress <ip adress> --port <port> --image_path <target image file path> --model_name alexnet
```


ex) densenet161
```
cd Server/densenet161_server/serve
torchserve --start --ncs --model-store model_store --models densenet161=densenet161.mar
cd <python rest api file>
python restapi.py --ip_adress <ip adress> --port <port> --image_path <target image file path> --model_name densenet161
```

# Javascript version(React) REST API
I used React for uploading image on browser and connect to rest api.

React demo is here(https://github.com/austingmhuang/upload_demo)

You have to install yarn.

Usage
```
git clone https://github.com/austingmhuang/upload_demo
cd upload-demo
yarn install
yarn start
```

if you upload image file, you get string which indicates what is in image.

