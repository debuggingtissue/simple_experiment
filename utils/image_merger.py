import sys
from PIL import Image, ImageDraw, ImageOps
import os
from os.path import isfile, join
import re
from pathlib import *


class ImageMerger:

    @staticmethod
    def create_overview_image(experiment_outputs_path, overview_output_path):
        path = Path(experiment_outputs_path)
        file_paths = [p for p in path.iterdir() if p.is_file()]
        order_dict = {}
        for file_path in file_paths:
            file_name = file_path.name
            order_dict[ImageMerger.find_first_number_in_string(file_name)] = file_path

        if not os.path.exists(overview_output_path):
            os.makedirs(overview_output_path)

        metrics_overview_image = ImageMerger.merge_images_vertically(order_dict)
        metrics_overview_image.save(f"{overview_output_path}/metrics_overview_{path.name}.png", 'PNG')

    @staticmethod
    def find_first_number_in_string(file_name):
        print(file_name)
        temp = re.findall(r'\d+', file_name)
        res = list(map(int, temp))
        return res[0]

    @staticmethod
    def merge_images_vertically(directory_of_image_paths):
        images = [Image.open(directory_of_image_paths[x]) for x in sorted(directory_of_image_paths)]
        widths, heights = zip(*(i.size for i in images))

        total_height = sum(heights)
        max_width = max(widths)

        new_im = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        return new_im

    @staticmethod
    def merge_images_horizontally(directory_of_image_paths):
        images = [Image.open(directory_of_image_paths[x]) for x in sorted(directory_of_image_paths)]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        return new_im
