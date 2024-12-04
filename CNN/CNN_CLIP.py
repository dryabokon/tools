import numpy as numpy
import cv2
import os
import torch
from tqdm import tqdm
from transformers import CLIPModel,CLIPProcessor
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
class CNN_CLIP():
    def __init__(self):
        self.name = 'CNN_CLIP'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return

# ---------------------------------------------------------------------------------------------------------------------
    def get_embedding(self,image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)
            embedding = image_emb.cpu().detach().numpy()[0]
        return embedding

# ---------------------------------------------------------------------------------------------------------------------
    def generate_embeddings(self, path_input, path_output,limit=1000000,mask = '*.png,*.jpg'):

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        #else:
            #tools_IO.remove_files(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            local_filenames = tools_IO.get_filenames(path_input + each,mask)[:limit]
            feature_filename = path_output + '/' + each + '_' + self.name + '.txt'
            embeddings, filenames = [], []

            if not os.path.isfile(feature_filename):
                for b, local_filename in tqdm(enumerate(local_filenames), total=len(local_filenames), desc=each):
                    image= cv2.imread(path_input + each + '/' + local_filename)
                    if image.shape[0]<32 or image.shape[1]<32:continue
                    #image = tools_image.desaturate(image)
                    if image is None:continue
                    embeddings.append(self.get_embedding(image))
                    filenames.append(local_filename)

                embeddings = numpy.array(embeddings)

                mat = numpy.zeros((embeddings.shape[0], embeddings.shape[1] + 1)).astype(str)
                mat[:, 0] = filenames
                mat[:, 1:] = embeddings
                tools_IO.save_mat(mat, feature_filename, fmt='%s', delim='\t')

        return
# ---------------------------------------------------------------------------------------------------------------------
