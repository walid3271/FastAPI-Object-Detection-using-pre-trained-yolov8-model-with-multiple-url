import json
from PIL import Image
from aiohttp import ClientSession
from io import BytesIO
import asyncio
from model import *

async def getImage(img_url, session):
    async with session.get(img_url) as response: #establishes an asynchronous context manager.
        img_data = await response.read() #await used to pause the execution until the reading operation completes.
        return BytesIO(img_data)
    

async def detection(model, img_content, confidence):
    img = Image.open(img_content)
    result = model(img, device='cpu', conf=confidence)
    detection = {}
    data = json.loads(result[0].tojson()) #converts the JSON-formatted result into a Python object.
    if len(data) == 0:
        res = {"AI": "No Detection"}
        detection.update(res)
    else:
        for item in data:
            obj_name = item['name']
            conf = item['confidence']
            box = item['box']
            res = {obj_name: {'Confidence': conf, 'Box': box}}
            detection.update(res)
    return detection


async def mainDet(url):
    async with ClientSession() as session: #ensures that an HTTP session is properly created and managed within an asynchronous environment.
        image = await asyncio.create_task(getImage(url, session))
        model_result = await asyncio.create_task(detection(Model, image, 0.3))
        result = json.dumps(model_result) #serializes the result into a JSON-formatted string.
        
        return result
