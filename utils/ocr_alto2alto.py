import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from chat_ocr_parsed_line import extract_text

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, TrOCRProcessor
from PIL import Image
import torch

processor = TrOCRProcessor.from_pretrained("raxtemur/trocr-base-ru")

# Load your fine-tuned model
model_path = "tridis"
model = VisionEncoderDecoderModel.from_pretrained(model_path)

# Load the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained("raxtemur/trocr-base-ru")

ns = {
    'alto': 'http://www.loc.gov/standards/alto/ns-v4#'
}

# Register the default namespace to avoid unnecessary prefixes
ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
ET.register_namespace('', 'http://www.loc.gov/standards/alto/ns-v4#')

folder = "1955"

xml_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.xml')]
print(xml_files)

for file in xml_files:
    filename = file.split(".")[0]
    result_path = os.path.join(f"{folder}/result", filename)
    print("Result path: ", result_path)
    if os.path.exists(result_path + "_with_text" + '.xml'):
        print(f"File {filename} already exists in the result folder.")
        continue
    try:
        # Load the XML file
        tree = ET.parse(f'{filename}.xml')
        root = tree.getroot()

        # Load the image
        image = cv2.imread(f'{filename}.jpg')

    except BaseException as e:
        print(e)
        print("File not found")
        exit()

    for page in root.findall('.//alto:Page', ns):
        n = 0
        previous_text = ""
        # Iterate through each TextBlock element
        for text_block in page.findall('.//alto:TextBlock', ns):
            # Iterate through each TextLine element within the TextBlock
            for text_line in text_block.findall('.//alto:TextLine', ns):
                coords = text_line.find('.//alto:Polygon', ns).attrib['POINTS']
                label = text_line.get('TAGREFS')
                if label == "LT16":
                    continue
                points = np.array([[int(n) for n in point.split(',')] for point in coords.split()], np.int32)
                points = points.reshape((-1, 1, 2))

                # Create a mask and perform the crop
                mask = np.zeros(image.shape[0:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points], (255))

                # Crop the image
                result = cv2.bitwise_and(image, image, mask=mask)
                x, y, w, h = cv2.boundingRect(points)  # Get the bounding box of the polygon
                cropped_image = result[y:y+h, x:x+w]

                # Save the cropped image
                try:
                    cv2.imwrite(f'{filename}_{n}_cropped_line.jpg', cropped_image)
                except BaseException as e:
                    print(e)
                    print("Error saving image")
                
                pixel_values = processor(cropped_image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(generated_text)
                                   
                try:
                    if generated_text:
                        string = text_line.find('.//alto:String', ns)
                        string.set('CONTENT', generated_text)
                except BaseException as e:
                    print(e)
                    print("Error extracting text")
    
    root.attrib['{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'] = (
        "http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
    )
    new_xml_file = f'{filename}.xml'.replace(folder, f"{folder}/result")
    tree.write(new_xml_file, encoding='utf-8', xml_declaration=True)

    print(f"Modified XML saved to {new_xml_file}")

                