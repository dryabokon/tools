import numpy as numpy
import cv2
import os
import progressbar
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
class CNN_RESNET50():
    def __init__(self):
        self.name = 'CNN_RESNET50'
        self.model = models.resnet50(pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        return

# ---------------------------------------------------------------------------------------------------------------------
    def get_embedding(self,image):
        pImage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self.preprocess(pImage).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image_tensor).squeeze()

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


