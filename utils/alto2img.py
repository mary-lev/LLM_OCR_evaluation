"""
Script Name: ALTO XML to Image and Text Processor
Author: [Your Name]
Date: [Current Date]

Description:
    This script processes ALTO XML files created by eScriptorium after running segmentation and recognition models,
    or through manual recognition by a human inside eScriptorium. It extracts text lines and their corresponding 
    polygonal regions from the associated images. The extracted text lines are saved as individual cropped images, 
    with the text content saved in separate text files. This output is ideal for training OCR models like TrOCR.

Usage:
    Run the script with the target folder containing ALTO XML and image files. The script will process all XML files 
    in the folder, outputting cropped images and text files in 'images' and 'lines' subfolders, respectively.

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - xml.etree.ElementTree
    - pathlib
    - logging

"""

import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Namespace for parsing the XML
NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}

def list_xml_files(folder: Path, xml_ext: str = ".xml") -> List[Path]:
    """List all XML files in the specified folder."""
    return list(folder.glob(f'*{xml_ext}'))

def load_image(filename: Path) -> Optional[np.ndarray]:
    """Load an image from the file system.

    Args:
        filename (Path): Path to the image file.

    Returns:
        Optional[np.ndarray]: Loaded image or None if the file is not found.
    """
    image = cv2.imread(str(filename))
    if image is None:
        logging.error(f"Image file not found: {filename}")
    return image

def parse_xml(file: Path) -> Optional[ET.Element]:
    """Parse the XML file and return the root element.
    """
    try:
        tree = ET.parse(file)
        return tree.getroot()
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file {file}: {e}")
        return None

def extract_text_lines(root: ET.Element) -> List[Tuple[np.ndarray, str]]:
    """Extract text lines and their associated polygons from the XML root.
    """
    text_lines = []
    for text_line in root.findall('.//alto:TextLine', NS):
        polygon_element = text_line.find('.//alto:Polygon', NS)
        if polygon_element is not None:
            coords = polygon_element.attrib['POINTS']
            label = text_line.get('TAGREFS')
            if label != "LT16":
                points = np.array([[int(coord) for coord in point.split(',')] for point in coords.split()], np.int32)
                points = points.reshape((-1, 1, 2))
                text = text_line.find('.//alto:String', NS).attrib.get('CONTENT', "")
                text_lines.append((points, text))
    return text_lines

def crop_image(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Crop the image around the given polygon.

    Args:
        image (np.ndarray): The full image from which to crop.
        polygon (np.ndarray): The polygon to define the crop area.

    Returns:
        np.ndarray: Cropped image.
    """
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    result = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(polygon)
    return result[y:y+h, x:x+w]

def save_image_and_text(cropped_image: np.ndarray, text: str, base_filename: Path, index: int, cropped_folder: Path, text_folder: Path) -> Dict[str, Path]:
    """Save the cropped image and associated text to their respective folders.

    Args:
        cropped_image (np.ndarray): The cropped image.
        text (str): The text associated with the cropped image.
        base_filename (Path): The base name of the original file.
        index (int): Index for the current line.
        cropped_folder (Path): Path to save cropped images.
        text_folder (Path): Path to save text files.

    Returns:
        Dict[str, Path]: Paths to the saved image and text files.
    """
    if not text:
        return {}
    cropped_img_filename = cropped_folder / f'{base_filename.stem}_{index}_cropped_line.jpg'
    txt_filename = text_folder / f'{base_filename.stem}_{index}_text.txt'

    cv2.imwrite(str(cropped_img_filename), cropped_image)
    with txt_filename.open("w") as f:
        f.write(text)

    return {"image": cropped_img_filename, "text": txt_filename}

def process_file(file: Path, cropped_folder: Path, text_folder: Path, image_ext: str = ".jpg") -> List[Dict[str, Path]]:
    """Process a single XML file: parse, extract text lines, crop images, and save results.

    Args:
        file (Path): Path to the XML file.
        cropped_folder (Path): Folder to save cropped images.
        text_folder (Path): Folder to save text files.
        image_ext (str): Extension of the image file.

    Returns:
        List[Dict[str, Path]]: List of paths to the saved images and text files.
    """
    base_filename = file.with_suffix('').with_name(file.stem.replace("_with_text", ""))
    image = load_image(base_filename.with_suffix(image_ext))
    if image is None:
        return []

    root = parse_xml(file)
    if root is None:
        return []

    results = []
    text_lines = extract_text_lines(root)
    for index, (polygon, text) in enumerate(text_lines):
        cropped_image = crop_image(image, polygon)
        result = save_image_and_text(cropped_image, text, base_filename, index, cropped_folder, text_folder)
        if result:
            results.append(result)

    return results

def main(folder: Path) -> List[Dict[str, Path]]:
    """Main function to process all XML files in the folder.

    Args:
        folder (Path): The folder containing XML files to process.

    Returns:
        List[Dict[str, Path]]: List of results for all processed files.
    """
    cropped_folder = folder / "images"
    text_folder = folder / "lines"
    cropped_folder.mkdir(parents=True, exist_ok=True)
    text_folder.mkdir(parents=True, exist_ok=True)

    xml_files = list_xml_files(folder)
    all_results = []

    for file in xml_files:
        logging.info(f"Processing file: {file}")
        results = process_file(file, cropped_folder, text_folder)
        all_results.extend(results)
        logging.info(f"Finished processing file: {file}")

    return all_results

if __name__ == "__main__":
    # Check if a folder name was provided as an argument
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        # Default to the current working directory if no argument is provided
        folder = Path.cwd()
    
    # Run the main function with the specified folder
    results = main(folder)
    
    # Print the results
    for result in results:
        logging.info(f"Processed {result['image']} and saved text to {result['text']}")
