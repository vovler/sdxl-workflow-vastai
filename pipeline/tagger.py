import csv
import os
import torch
import numpy as np
from PIL import Image

import pipeline.models as models
import pipeline.defaults as defaults

class WD14Tagger:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = models.WDTaggerONNX(defaults.WD14_TAGGER_MODEL_PATH, self.device)
        self.tags = self.load_tags()

    def load_tags(self):
        tags_path = defaults.WD14_TAGGER_TAGS_PATH
        with open(tags_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            tags = [row[0] for row in reader]
        return tags

    def preprocess_image(self, image: Image.Image):
        target_size = self.model.image_size
        
        # Resize
        ratio = float(target_size) / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)

        # Pad to square
        square_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        square_image.paste(image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))

        # To numpy and preprocess
        image_np = np.array(square_image).astype(np.float32)
        image_np = image_np[:, :, ::-1]  # RGB -> BGR
        image_np = np.expand_dims(image_np, axis=0) # Add batch dimension

        return torch.from_numpy(image_np.copy()).permute(0, 3, 1, 2)  # NHWC -> NCHW

    def filter_image(self, image: Image.Image, threshold: float = 0.35, blacklist: list = None):
        if not os.path.exists(defaults.WD14_TAGGER_MODEL_PATH):
            print("WD14 Tagger model not found. Skipping filter.")
            return False, []

        if blacklist is None:
            blacklist = defaults.WD14_TAGGER_BLACKLIST

        preprocessed_image = self.preprocess_image(image).to(self.device)
        
        probs = self.model(preprocessed_image).cpu().numpy()[0]
        
        # Normalize blacklist for comparison
        processed_blacklist = {tag.lower().replace("_", " ") for tag in blacklist}
        
        detected_blacklisted_tags = []
        for i, tag in enumerate(self.tags):
            prob = probs[i]
            processed_tag = tag.lower().replace("_", " ")
            if prob > threshold and processed_tag in processed_blacklist:
                detected_blacklisted_tags.append(tag)
        
        is_nsfw = len(detected_blacklisted_tags) > 0
        return is_nsfw, detected_blacklisted_tags 