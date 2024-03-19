import numpy as numpy
import cv2
import os
import progressbar
import torch
from transformers import AutoImageProcessor, ViTModel
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
class CNN_VIT():
    def __init__(self):
        self.name = 'CNN_VIT'

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        return

# ---------------------------------------------------------------------------------------------------------------------
    def get_embedding(self,image):
        inputs = self.image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state
        return embedding
# ---------------------------------------------------------------------------------------------------------------------
    def generate_embeddings(self, path_input, path_output,limit=1000000,mask = '*.png,*.jpg'):

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        #else:
            #tools_IO.remove_files(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = tools_IO.get_filenames(path_input + each,mask)[:limit]
            feature_filename = path_output + '/' + each + '_' + self.name + '.txt'
            embeddings, filenames = [], []

            if not os.path.isfile(feature_filename):
                bar = progressbar.ProgressBar(maxval=len(local_filenames))
                bar.start()
                for b, local_filename in enumerate(local_filenames):
                    bar.update(b)
                    image= cv2.imread(path_input + each + '/' + local_filename)
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
    def predict(self,image):

        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        res = self.model(**inputs)

        return
# ---------------------------------------------------------------------------------------------------------------------


