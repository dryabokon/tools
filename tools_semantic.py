import cv2
import os
import numpy
import pandas as pd
from os import listdir
import fnmatch
import requests
from PIL import Image as PillowImage
import urllib.request
import re
import uuid
from scenedetect import detect, ContentDetector
import torch
# ----------------------------------------------------------------------------------------------------------------------
import sys

sys.path.append('./tools')
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_image
import tools_tensor_view
import tools_video


# ----------------------------------------------------------------------------------------------------------------------
class Semantic_proc:
    def __init__(self, folder_out, cold_start=True, clip_model_make='OPENAI'):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.clip_model_make = clip_model_make
        self.folder_out = folder_out
        self.hex_mode = False
        self.token_size = 512
        self.started = False
        self.df_images = None
        self.df_words = None
        if cold_start:
            self.cold_start()
            self.started = True

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def cold_start(self):
        if not self.started:
            if self.clip_model_make == 'huggingface':
                from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
                self.model_id = "openai/clip-vit-base-patch32"
                self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_id)
                self.model = CLIPModel.from_pretrained(self.model_id)
                self.processor = CLIPProcessor.from_pretrained(self.model_id)
            else:
                import clip
                self.model, self.processor = clip.load("ViT-B/32", device=self.device)
                self.tokenizer = clip.tokenize

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def get_filenames(self, path_input, list_of_masks):
        local_filenames = []
        for mask in list_of_masks.split(','):
            res = listdir(path_input)
            if mask != '*.*':
                res = fnmatch.filter(res, mask)
            local_filenames += res

        return numpy.sort(numpy.array(local_filenames))

    # ----------------------------------------------------------------------------------------------------------------------
    def get_token_text(self, words):
        if self.clip_model_make == 'huggingface':
            inputs = self.tokenizer(words, return_tensors="pt", padding=True)
            text_emb = self.model.get_text_features(**inputs).to(self.device).cpu().detach().numpy()
        else:
            with torch.no_grad():
                inputs = self.tokenizer(words)
                text_emb = self.model.encode_text(inputs.to(self.device)).cpu().detach().numpy()

        df = pd.DataFrame(text_emb)

        return df

    # ----------------------------------------------------------------------------------------------------------------------
    def get_token_images(self, list_of_images):
        if self.clip_model_make == 'huggingface':
            inputs = self.processor(text=None, images=list_of_images, return_tensors='pt', padding=True)['pixel_values']
            img_emb = self.model.get_image_features(inputs).to(self.device).cpu().detach().numpy()
        else:
            with torch.no_grad():
                img_emb = [self.model.encode_image(self.processor(PillowImage.fromarray(image)).unsqueeze(0).to(self.device)) for image in list_of_images]
                img_emb = [torch.flatten(t).cpu().detach().numpy() for t in img_emb]

        df = pd.DataFrame(img_emb)
        return df

    # ----------------------------------------------------------------------------------------------------------------------
    def tokenize_images(self, folder_images, mask='*.png,*.jpg,*.jpeg', batch_size=100):

        filenames_images = self.get_filenames(folder_images, mask)
        print('%d scenes found' % len(filenames_images))

        df = pd.DataFrame([])
        batch_iter = 0
        while batch_iter < len(filenames_images):
            batch_filenames = filenames_images[batch_iter:batch_iter + batch_size]
            list_of_images = [cv2.imread(folder_images + filename) for filename in batch_filenames]
            df = self.get_token_images(list_of_images)
            if self.hex_mode:
                df = tools_DF.to_hex(df)

            df = pd.concat([pd.DataFrame({'image': batch_filenames}), df], axis=1)

            if batch_iter == 0:
                mode, header = 'w+', True
            else:
                mode, header = 'a', False

            df.to_csv(self.folder_out + 'tokens_%s_%s.csv' % (self.clip_model_make, folder_images.split('/')[-2]),
                      index=False, float_format='%.4f', mode=mode, header=header)
            batch_iter += batch_size

        return df

    # ----------------------------------------------------------------------------------------------------------------------
    def tokenize_URLs_images(self, URLs, captions=None, do_save=True):

        if captions is None:
            captions = [''] * len(URLs)

        filename_out = self.folder_out + 'tokens_%s.csv' % self.clip_model_make
        if os.path.isfile(filename_out):
            os.remove(filename_out)

        for i, (URL, caption) in enumerate(zip(URLs, captions)):
            filename_image = '%06d.jpg' % i
            try:
                response = requests.get(URL, stream=True, timeout=2, allow_redirects=False)
            except:
                print(i, 'Timeout ', URL)
                continue

            if not response.ok:
                print(i, 'Bad response ', URL)
                continue

            try:
                image = cv2.cvtColor(numpy.array(Image.open(response.raw)), cv2.COLOR_RGB2BGR)
            except:
                print(i, 'Bad payload ', URL)
                continue
            if do_save:
                cv2.imwrite(self.folder_out + filename_image, image)

            print(i, 'OK ', URL)

            df = self.get_token_images([image])
            if self.hex_mode:
                df = tools_DF.to_hex(df)

            df = pd.concat([pd.DataFrame({'image': [filename_image]}), df, pd.DataFrame({'caption': [caption]})],
                           axis=1)

            if os.path.isfile(filename_out):
                mode, header = 'a', False
            else:
                mode, header = 'w+', True

            df.to_csv(filename_out, index=False, float_format='%.4f', mode=mode, header=header)

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def tokenize_video_scenes(self, filename_in, folder_out, prefix=''):
        if not os.path.isfile(filename_in):
            df_log = pd.DataFrame({'filename': []})
        else:
            scene_list = detect(filename_in, ContentDetector())
            frame_IDs = [(scene[0].get_frames() + scene[1].get_frames()) // 2 for scene in scene_list]
            tools_video.extract_frames_v3(filename_in, folder_out, frame_IDs=frame_IDs, prefix=prefix)

            df_log = pd.DataFrame({'filename': [prefix + '%06d.jpg' % frame_ID for frame_ID in frame_IDs]})
        return df_log

    # ----------------------------------------------------------------------------------------------------------------------
    def tokenize_words(self, filename_words, batch_size=50):
        words = [w for w in pd.read_csv(filename_words, header=None, sep='\t').values[:, 0]]

        batch_iter = 0
        while batch_iter < len(words):
            batch_words = words[batch_iter:batch_iter + batch_size]
            df = self.get_token_text(batch_words)

            if self.hex_mode:
                df = tools_DF.to_hex(df)

            df = pd.concat([pd.DataFrame({'words': batch_words}), df], axis=1)
            mode = 'w+' if batch_iter == 0 else 'a'
            header = True if batch_iter == 0 else False
            df.to_csv(self.folder_out + 'tokens_%s_%s.csv' % (
            self.clip_model_make, filename_words.split('/')[-1].split('.')[0]), index=False, float_format='%.4f',
                      mode=mode, header=header)
            batch_iter += batch_size

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def compose_thumbnails(self, folder_in, filenames_images, fiename_out):

        small_width, small_height = 160, 120
        tensor = []
        for filename in filenames_images:
            if os.path.isfile(folder_in + filename):
                image = tools_image.smart_resize(cv2.imread(folder_in + filename), small_height, small_width)
            else:
                image = numpy.full((small_height, small_width, 3), 128, dtype=numpy.uint8)
            tensor.append(image)

        image = tools_tensor_view.tensor_color_4D_to_image(numpy.transpose(numpy.array(tensor), (1, 2, 3, 0)))
        cv2.imwrite(self.folder_out + fiename_out, image)

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def get_top_items(self, items, confidences, top_n=3):
        t_df = pd.DataFrame({'items': items, 'conf': confidences})
        t_df = t_df.sort_values(by=t_df.columns[1], ascending=False)[:top_n]
        items = [i for i in t_df.iloc[:, 0]]
        confidences = [i for i in t_df.iloc[:, 1]]
        return items, confidences

    # ----------------------------------------------------------------------------------------------------------------------
    def similarity_to_description(self, df, top_n=3):

        with open(self.folder_out + "descript.ion", mode='w+') as f_handle:
            for r in range(df.shape[0]):
                items, confidences = self.get_top_items(df.columns[1:], df.iloc[r, 1:], top_n=top_n)
                str_items = ' '.join(
                    [item + '(%d)' % (100 * confidence) for item, confidence in zip(items, confidences)])

                f_handle.write("%s %s\n" % (df.iloc[r, 0], str_items))
            f_handle.close()

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def preprocess_tokens(self, df_tokens):
        if isinstance(df_tokens, str):
            df = pd.read_csv(df_tokens)
        else:
            df = df_tokens

        df_temp = df.iloc[:, 1:1 + self.token_size]

        if self.hex_mode:
            df_temp = tools_DF.from_hex(df_temp)


        df_temp = df_temp / numpy.linalg.norm(df_temp, axis=0)
        df = pd.concat([df.iloc[:, 0], df_temp], axis=1)

        return df

    # ----------------------------------------------------------------------------------------------------------------------
    def tokens_similarity(self, filename_tokens1, filename_tokens2, top_n=3):

        df1 = self.preprocess_tokens(filename_tokens1)
        df2 = self.preprocess_tokens(filename_tokens2)
        names1 = df1.iloc[:, 0].copy()
        df_similarity = df2.iloc[:, 0].copy()

        df1 = df1.iloc[:, 1:1 + self.token_size]
        df2 = df2.iloc[:, 1:1 + self.token_size]

        for index, row in df1.iterrows():
            df_similarity = pd.concat([df_similarity, df2.dot(row).rename(names1[index])], axis=1)

        # df_similarity.to_csv(self.folder_out + 'similarity.csv', index=False, float_format='%.2f')

        self.similarity_to_description(df_similarity, top_n=top_n)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def search_images(self, query_text, filename_tokens_images, filename_tokens_words=None, top_n=5):

        if self.df_images is None:
            self.df_images = self.preprocess_tokens(filename_tokens_images)

        if self.df_words is None:
            self.df_words = self.preprocess_tokens(filename_tokens_words)

        df_words_f = tools_DF.apply_filter(self.df_words, self.df_words.columns[0], query_text)
        if df_words_f.shape[0] == 0:
            df_words_f = pd.concat([pd.DataFrame({'words': [query_text]}), self.get_token_text(query_text)], axis=1)


        row = pd.Series(df_words_f.iloc[0, 1:1 + self.token_size], name=query_text)
        mat = self.df_images.iloc[:, 1:1 + self.token_size]
        # df_similarity = pd.DataFrame({'P': mat.dot(row.values).values}, index=self.df_images.iloc[:, 0]).T

        with torch.no_grad():
            t = torch.matmul(torch.tensor(mat.values).float(), torch.tensor(row).float())
            df_similarity = pd.DataFrame(t.cpu().detach().numpy(), index=self.df_images.iloc[:, 0]).T

        df_similarity['text'] = query_text
        df_similarity = pd.concat([df_similarity.iloc[:, -1], df_similarity.iloc[:, :-1]], axis=1)

        # df_similarity.to_csv(self.folder_out + 'similarity.csv', index=False, float_format='%.2f')
        filenames_images, confidences = self.get_top_items(df_similarity.columns[1:], df_similarity.iloc[0, 1:],
                                                           top_n=top_n)

        return filenames_images

    # ----------------------------------------------------------------------------------------------------------------------
    def tokenize_youtube_scenes(self, URLs, filename_log):
        grab_resolution = '360p'
        max_scenes = 10
        filename_tmp_video = uuid.uuid3(uuid.NAMESPACE_URL, 'delme').hex + '.mp4'

        if os.path.isfile(self.folder_out + filename_log):
            df_log = pd.read_csv(self.folder_out + filename_log)
        else:
            df_log = pd.DataFrame([])

        for URL in numpy.unique(URLs):
            if (df_log.shape[0] > 0) and (URL in df_log['URL']): continue
            prefix = uuid.uuid3(uuid.NAMESPACE_URL, URL).hex + '_'

            df_log1 = tools_video.grab_youtube_video(URL, self.folder_out, filename_tmp_video,
                                                     resolution=grab_resolution)

            if not os.path.isfile(self.folder_out + filename_tmp_video):
                continue
            frame_IDs = [(scene[0].get_frames() + scene[1].get_frames()) // 2 for scene in
                         detect(self.folder_out + filename_tmp_video, ContentDetector())]
            tools_video.extract_frames_v3(self.folder_out + filename_tmp_video, self.folder_out, frame_IDs[:max_scenes],
                                          prefix=prefix, scale=1)
            df_log2 = pd.DataFrame({'image': self.get_filenames(self.folder_out, prefix + '*.jpg')})
            # df_log2 = self.tokenize_images(self.folder_out,mask=prefix+'*.jpg')

            df_log = df_log2.merge(df_log1, how='cross')

            if os.path.isfile(self.folder_out + filename_log):
                header, mode = False, 'a'
            else:
                header, mode = True, 'w'
            df_log.to_csv(self.folder_out + filename_log, index=False, header=header, mode=mode)
            tools_video.rescale_overwrite_images(self.folder_out, mask=prefix + '*.jpg', target_width=360)
            if os.path.isfile(self.folder_out + filename_tmp_video): os.remove(self.folder_out + filename_tmp_video)

            print('%d scenes %s %s' % (df_log2.shape[0], URL, df_log1['title'].iloc[0]))
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def get_ULRs(self, queries, limit=3):
        res = []
        if not isinstance(queries, list):
            queries = [queries]

        for query in queries:
            query = query.replace(' ', '+')
            html = urllib.request.urlopen("https://www.youtube.com/results?search_query=%s" % query)
            video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
            res += ["https://www.youtube.com/watch?v=" + x for x in video_ids[:limit]]
        return res
# ---------------------------------------------------------------------------------------------------------------------