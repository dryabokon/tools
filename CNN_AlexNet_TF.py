import numpy as numpy
import tensorflow as tf
import cv2
import os
from os import listdir
import fnmatch
import bcolz
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
net_data = numpy.load("../_weights/bvlc_alexnet.npy", encoding="latin1").item()
# ----------------------------------------------------------------------------------------------------------------------
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
# ----------------------------------------------------------------------------------------------------------------------
class CNN_AlexNet_TF():
    def __init__(self):
        self.name = 'CNN_AlexNet_TF'
        self.input_shape = (227, 227)
        self.nb_classes = 4096
        self.x = tf.placeholder(tf.float32, (None, self.input_shape[0], self.input_shape[1], 3))
        self.input_placeholder = tf.image.resize_images(self.x, (227, 227))

        self.class_names = class_names
        self.build()

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def build(self):
        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')

        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1_in = conv(self.input_placeholder, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
        self.layer_conv1 = conv1_in
        self.conv1W = conv1W

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        self.layer_maxpool1 = maxpool1

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5
        k_w = 5
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        self.layer_conv2 = conv2_in
        self.conv2W = conv2W

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        self.layer_maxpool2 = maxpool2

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 1
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        self.layer_conv3 = conv3_in
        self.conv3W = conv3W


        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 2
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        self.layer_conv4 = conv4_in
        self.conv4W = conv4W


        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3
        k_w = 3
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        self.layer_conv5 = conv5_in
        self.conv5W = conv5W

        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        self.layer_maxpool5 = maxpool5

        # fc6, 4096
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        flat5 = tf.reshape(maxpool5, [-1, int(numpy.prod(maxpool5.get_shape()[1:]))])
        fc6 = tf.nn.relu(tf.matmul(flat5, fc6W) + fc6b)
        self.layer_fc6 = fc6

        # fc7, 4096
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc7 = tf.nn.relu(tf.matmul(fc6, fc7W) + fc7b)
        self.layer_fc7 = fc7

        # fc8, 4096
        shape = (fc7.get_shape().as_list()[-1], self.nb_classes)
        fc8W_features = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
        fc8b_features = tf.Variable(tf.zeros(self.nb_classes))
        logits = tf.matmul(fc7, fc8W_features) + fc8b_features
        self.layer_feature= tf.nn.softmax(logits)

        # fc8, 1000
        fc8W_classes = tf.Variable(net_data["fc8"][0])
        fc8b_classes = tf.Variable(net_data["fc8"][1])
        logits = tf.matmul(fc7, fc8W_classes) + fc8b_classes
        self.layer_classes = tf.nn.softmax(logits)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def generate_features(self, path_input, path_output,limit=1000000,mask='*.png'):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        else:
            tools_IO.remove_files(path_output)

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            feature_filename = path_output + each + '_features.txt'
            features = []

            if not os.path.isfile(feature_filename):
                for i in range (0,local_filenames.shape[0]):
                    image= cv2.imread(path_input + each + '/' + local_filenames[i])
                    image = cv2.resize(image,(self.input_shape[0],self.input_shape[1]))
                    feature = sess.run(self.layer_feature, feed_dict={self.x: [image]})
                    features.append(feature[0])

                features = numpy.array(features)

                if self.src=='bcolz':
                    if not os.path.exists(path_output + each):os.makedirs(path_output + each)
                    bcolz.carray(local_filenames, rootdir=(path_output + each + '/' + 'filenames.bcolz'),mode='w').flush()
                    bcolz.carray(features,rootdir=(path_output + each + '/' + 'features.bcolz'), mode='w').flush()

                else:
                    mat = numpy.zeros((features.shape[0], features.shape[1] + 1)).astype(numpy.str)
                    mat[:, 0] = local_filenames
                    mat[:, 1:] = features
                    tools_IO.save_mat(mat, feature_filename, fmt='%s', delim='\t')
        sess.close()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def predict_classes(self, path_input, filename_output, limit=1000000,mask='*.png'):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        #writer = tf.summary.FileWriter('../_images/output/', sess.graph)
        #writer.close()

        patterns = numpy.sort(numpy.array([f.path[len(path_input):] for f in os.scandir(path_input) if f.is_dir()]))

        for each in patterns:
            print(each)
            local_filenames = numpy.array(fnmatch.filter(listdir(path_input + each), mask))[:limit]
            for i in range(0, local_filenames.shape[0]):
                image = cv2.imread(path_input + each + '/' + local_filenames[i])
                image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                prob = sess.run(self.layer_classes, feed_dict={self.x: [image]})
                prob = prob[0]
                idx = numpy.argsort(-prob)[0]
                label = self.class_names[idx]

                tools_IO.save_labels(path_input+each+'/'+filename_output, numpy.array([local_filenames[i]]), numpy.array([label]), append=i, delim=' ')


        sess.close()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def tensor_gray_3D_to_image(self, tensor, rows=8):
        h, w, R, C = tensor.shape[0], tensor.shape[1], rows, int(tensor.shape[2] / rows)
        image = numpy.zeros((h * R, w * C), dtype=numpy.float32)
        for i in range(0, tensor.shape[2]):
            col, row = i % C, int(i / C)
            image[h * row:h * row + h, w * col:w * col + w] = tensor[:, :, i]
        return image
# ---------------------------------------------------------------------------------------------------------------------
    def tensor_color_4D_to_image(self, tensor, rows=8):
        h, w, R, C = tensor.shape[0], tensor.shape[1], rows, int(tensor.shape[3] / rows)
        image = numpy.zeros((h * R,w * C, tensor.shape[2]),dtype=numpy.float32)
        for i in range(0, tensor.shape[3]):
            col, row = i % C, int(i / C)
            image[h * row:h * row + h, w * col:w * col + w, :] = tensor[:, :, :, i]
        return image
# ---------------------------------------------------------------------------------------------------------------------
    def tensor_gray_4D_to_image(self, tensor, rows = 8, sub_rows=8):

        sub_image = self.tensor_gray_3D_to_image(tensor[:, :, :, 0], rows=sub_rows)
        h, w = sub_image.shape[0], sub_image.shape[1]
        R = rows
        C = int(tensor.shape[3] / rows)

        image = numpy.zeros((h * R, w * C), dtype=numpy.float32)
        for i in range (0,tensor.shape[3]):
            col, row = i % C, int(i / C)
            sub_image = self.tensor_gray_3D_to_image(tensor[:,:,:,i], rows=sub_rows)
            image[h * row:h * row + h, w * col:w * col + w] = sub_image[:, :]

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def visualize_filters(self, path_output):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        tensor = 255 * (0.5 + self.conv1W.eval(sess)) #(11,11,3,96)
        cv2.imwrite(path_output + 'filter1.png', self.tensor_color_4D_to_image(tensor,rows=8))

        tensor = 255 * (0.5 + self.conv2W.eval(sess)) #(5,5,48,256)
        cv2.imwrite(path_output + 'filter2.png', self.tensor_gray_4D_to_image(tensor, sub_rows=8, rows=16, ))

        tensor = 255 * (0.5 + self.conv3W.eval(sess)) #(3,3,256,384)
        cv2.imwrite(path_output + 'filter3.png', self.tensor_gray_4D_to_image(tensor, sub_rows=16, rows=12))

        tensor = 255 * (0.5 + self.conv4W.eval(sess)) #(3,3,192,384)
        cv2.imwrite(path_output + 'filter4.png', self.tensor_gray_4D_to_image(tensor, sub_rows=12, rows=12))

        tensor = 255 * (0.5 + self.conv5W.eval(sess)) #(3,3,192,256)
        cv2.imwrite(path_output + 'filter5.png', self.tensor_gray_4D_to_image(tensor, sub_rows=12, rows=16))
        sess.close()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def visualize_layers(self, filename_input, path_output):

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        image = cv2.resize(cv2.imread(filename_input), (self.input_shape[0], self.input_shape[1]))

        output = sess.run(self.layer_conv1, feed_dict={self.x: [image]})[0] #(57,57,96)
        cv2.imwrite(path_output + 'layer01_conv1.png',self.tensor_gray_3D_to_image(output,rows=8))

        output = sess.run(self.layer_maxpool1, feed_dict={self.x: [image]})[0] #(28,28,96)
        cv2.imwrite(path_output + 'layer02_pool1.png', self.tensor_gray_3D_to_image(2*output,rows=8))

        output = sess.run(self.layer_conv2, feed_dict={self.x: [image]})[0] #(28,28,256)
        cv2.imwrite(path_output + 'layer03_conv2.png', self.tensor_gray_3D_to_image(output,rows=16))

        output = sess.run(self.layer_maxpool2, feed_dict={self.x: [image]})[0] #(13,13,256)
        cv2.imwrite(path_output + 'layer04_pool2.png', self.tensor_gray_3D_to_image(output,rows=16))

        output = sess.run(self.layer_conv3, feed_dict={self.x: [image]})[0]  #(13,13,384)
        cv2.imwrite(path_output + 'layer04_conv3.png', self.tensor_gray_3D_to_image(output,rows=16))

        output = sess.run(self.layer_conv4, feed_dict={self.x: [image]})[0] #(13,13,384)
        cv2.imwrite(path_output + 'layer05_conv4.png', self.tensor_gray_3D_to_image(output,rows=16))

        output = sess.run(self.layer_conv5, feed_dict={self.x: [image]})[0] #(13,13,256)
        cv2.imwrite(path_output + 'layer06_conv5.png', self.tensor_gray_3D_to_image(output,rows=16))

        output = sess.run(self.layer_maxpool5, feed_dict={self.x: [image]})[0] #(6,6,256)
        cv2.imwrite(path_output + 'layer07_pool5.png', self.tensor_gray_3D_to_image(2*output,rows=16))

        output = sess.run(self.layer_fc6, feed_dict={self.x: [image]})[0] #4096
        cv2.imwrite(path_output + 'layer08_fc6.png', (output*255/numpy.max(output)).reshape(64,-1))

        output = sess.run(self.layer_fc7, feed_dict={self.x: [image]})[0] #4096
        cv2.imwrite(path_output + 'layer09_fc7.png', (output*255/numpy.max(output)).reshape(64,-1))

        output = sess.run(self.layer_feature, feed_dict={self.x: [image]})[0] #4096
        cv2.imwrite(path_output + 'layer10_features.png', (output*255/numpy.max(output)).reshape(64,-1))

        output = sess.run(self.layer_classes, feed_dict={self.x: [image]})[0] #1000
        cv2.imwrite(path_output + 'layer11_classed.png', (output*255/numpy.max(output)).reshape(50,-1))

        idx = numpy.argsort(-output)
        mat = numpy.array([output[idx],numpy.array(self.class_names)[idx]]).T
        tools_IO.save_mat(mat,path_output+'predictions.txt',fmt='%s',delim='\t')

        sess.close()
        return
# ---------------------------------------------------------------------------------------------------------------------

class_names =  '''tench, Tinca tinca
goldfish, Carassius auratus
great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
tiger shark, Galeocerdo cuvieri
hammerhead, hammerhead shark
electric ray, crampfish, numbfish, torpedo
stingray
cock
hen
ostrich, Struthio camelus
brambling, Fringilla montifringilla
goldfinch, Carduelis carduelis
house finch, linnet, Carpodacus mexicanus
junco, snowbird
indigo bunting, indigo finch, indigo bird, Passerina cyanea
robin, American robin, Turdus migratorius
bulbul
jay
magpie
chickadee
water ouzel, dipper
kite
bald eagle, American eagle, Haliaeetus leucocephalus
vulture
great grey owl, great gray owl, Strix nebulosa
European fire salamander, Salamandra salamandra
common newt, Triturus vulgaris
eft
spotted salamander, Ambystoma maculatum
axolotl, mud puppy, Ambystoma mexicanum
bullfrog, Rana catesbeiana
tree frog, tree-frog
tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
loggerhead, loggerhead turtle, Caretta caretta
leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea
mud turtle
terrapin
box turtle, box tortoise
banded gecko
common iguana, iguana, Iguana iguana
American chameleon, anole, Anolis carolinensis
whiptail, whiptail lizard
agama
frilled lizard, Chlamydosaurus kingi
alligator lizard
Gila monster, Heloderma suspectum
green lizard, Lacerta viridis
African chameleon, Chamaeleo chamaeleon
Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis
African crocodile, Nile crocodile, Crocodylus niloticus
American alligator, Alligator mississipiensis
triceratops
thunder snake, worm snake, Carphophis amoenus
ringneck snake, ring-necked snake, ring snake
hognose snake, puff adder, sand viper
green snake, grass snake
king snake, kingsnake
garter snake, grass snake
water snake
vine snake
night snake, Hypsiglena torquata
boa constrictor, Constrictor constrictor
rock python, rock snake, Python sebae
Indian cobra, Naja naja
green mamba
sea snake
horned viper, cerastes, sand viper, horned asp, Cerastes cornutus
diamondback, diamondback rattlesnake, Crotalus adamanteus
sidewinder, horned rattlesnake, Crotalus cerastes
trilobite
harvestman, daddy longlegs, Phalangium opilio
scorpion
black and gold garden spider, Argiope aurantia
barn spider, Araneus cavaticus
garden spider, Aranea diademata
black widow, Latrodectus mactans
tarantula
wolf spider, hunting spider
tick
centipede
black grouse
ptarmigan
ruffed grouse, partridge, Bonasa umbellus
prairie chicken, prairie grouse, prairie fowl
peacock
quail
partridge
African grey, African gray, Psittacus erithacus
macaw
sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
lorikeet
coucal
bee eater
hornbill
hummingbird
jacamar
toucan
drake
red-breasted merganser, Mergus serrator
goose
black swan, Cygnus atratus
tusker
echidna, spiny anteater, anteater
platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus
wallaby, brush kangaroo
koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus
wombat
jellyfish
sea anemone, anemone
brain coral
flatworm, platyhelminth
nematode, nematode worm, roundworm
conch
snail
slug
sea slug, nudibranch
chiton, coat-of-mail shell, sea cradle, polyplacophore
chambered nautilus, pearly nautilus, nautilus
Dungeness crab, Cancer magister
rock crab, Cancer irroratus
fiddler crab
king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica
American lobster, Northern lobster, Maine lobster, Homarus americanus
spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish
crayfish, crawfish, crawdad, crawdaddy
hermit crab
isopod
white stork, Ciconia ciconia
black stork, Ciconia nigra
spoonbill
flamingo
little blue heron, Egretta caerulea
American egret, great white heron, Egretta albus
bittern
crane
limpkin, Aramus pictus
European gallinule, Porphyrio porphyrio
American coot, marsh hen, mud hen, water hen, Fulica americana
bustard
ruddy turnstone, Arenaria interpres
red-backed sandpiper, dunlin, Erolia alpina
redshank, Tringa totanus
dowitcher
oystercatcher, oyster catcher
pelican
king penguin, Aptenodytes patagonica
albatross, mollymawk
grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus
killer whale, killer, orca, grampus, sea wolf, Orcinus orca
dugong, Dugong dugon
sea lion
Chihuahua
Japanese spaniel
Maltese dog, Maltese terrier, Maltese
Pekinese, Pekingese, Peke
Shih-Tzu
Blenheim spaniel
papillon
toy terrier
Rhodesian ridgeback
Afghan hound, Afghan
basset, basset hound
beagle
bloodhound, sleuthhound
bluetick
black-and-tan coonhound
Walker hound, Walker foxhound
English foxhound
redbone
borzoi, Russian wolfhound
Irish wolfhound
Italian greyhound
whippet
Ibizan hound, Ibizan Podenco
Norwegian elkhound, elkhound
otterhound, otter hound
Saluki, gazelle hound
Scottish deerhound, deerhound
Weimaraner
Staffordshire bullterrier, Staffordshire bull terrier
American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
Bedlington terrier
Border terrier
Kerry blue terrier
Irish terrier
Norfolk terrier
Norwich terrier
Yorkshire terrier
wire-haired fox terrier
Lakeland terrier
Sealyham terrier, Sealyham
Airedale, Airedale terrier
cairn, cairn terrier
Australian terrier
Dandie Dinmont, Dandie Dinmont terrier
Boston bull, Boston terrier
miniature schnauzer
giant schnauzer
standard schnauzer
Scotch terrier, Scottish terrier, Scottie
Tibetan terrier, chrysanthemum dog
silky terrier, Sydney silky
soft-coated wheaten terrier
West Highland white terrier
Lhasa, Lhasa apso
flat-coated retriever
curly-coated retriever
golden retriever
Labrador retriever
Chesapeake Bay retriever
German short-haired pointer
vizsla, Hungarian pointer
English setter
Irish setter, red setter
Gordon setter
Brittany spaniel
clumber, clumber spaniel
English springer, English springer spaniel
Welsh springer spaniel
cocker spaniel, English cocker spaniel, cocker
Sussex spaniel
Irish water spaniel
kuvasz
schipperke
groenendael
malinois
briard
kelpie
komondor
Old English sheepdog, bobtail
Shetland sheepdog, Shetland sheep dog, Shetland
collie
Border collie
Bouvier des Flandres, Bouviers des Flandres
Rottweiler
German shepherd, German shepherd dog, German police dog, alsatian
Doberman, Doberman pinscher
miniature pinscher
Greater Swiss Mountain dog
Bernese mountain dog
Appenzeller
EntleBucher
boxer
bull mastiff
Tibetan mastiff
French bulldog
Great Dane
Saint Bernard, St Bernard
Eskimo dog, husky
malamute, malemute, Alaskan malamute
Siberian husky
dalmatian, coach dog, carriage dog
affenpinscher, monkey pinscher, monkey dog
basenji
pug, pug-dog
Leonberg
Newfoundland, Newfoundland dog
Great Pyrenees
Samoyed, Samoyede
Pomeranian
chow, chow chow
keeshond
Brabancon griffon
Pembroke, Pembroke Welsh corgi
Cardigan, Cardigan Welsh corgi
toy poodle
miniature poodle
standard poodle
Mexican hairless
timber wolf, grey wolf, gray wolf, Canis lupus
white wolf, Arctic wolf, Canis lupus tundrarum
red wolf, maned wolf, Canis rufus, Canis niger
coyote, prairie wolf, brush wolf, Canis latrans
dingo, warrigal, warragal, Canis dingo
dhole, Cuon alpinus
African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus
hyena, hyaena
red fox, Vulpes vulpes
kit fox, Vulpes macrotis
Arctic fox, white fox, Alopex lagopus
grey fox, gray fox, Urocyon cinereoargenteus
tabby, tabby cat
tiger cat
Persian cat
Siamese cat, Siamese
Egyptian cat
cougar, puma, catamount, mountain lion, painter, panther, Felis concolor
lynx, catamount
leopard, Panthera pardus
snow leopard, ounce, Panthera uncia
jaguar, panther, Panthera onca, Felis onca
lion, king of beasts, Panthera leo
tiger, Panthera tigris
cheetah, chetah, Acinonyx jubatus
brown bear, bruin, Ursus arctos
American black bear, black bear, Ursus americanus, Euarctos americanus
ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus
sloth bear, Melursus ursinus, Ursus ursinus
mongoose
meerkat, mierkat
tiger beetle
ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle
ground beetle, carabid beetle
long-horned beetle, longicorn, longicorn beetle
leaf beetle, chrysomelid
dung beetle
rhinoceros beetle
weevil
fly
bee
ant, emmet, pismire
grasshopper, hopper
cricket
walking stick, walkingstick, stick insect
cockroach, roach
mantis, mantid
cicada, cicala
leafhopper
lacewing, lacewing fly
dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk
damselfly
admiral
ringlet, ringlet butterfly
monarch, monarch butterfly, milkweed butterfly, Danaus plexippus
cabbage butterfly
sulphur butterfly, sulfur butterfly
lycaenid, lycaenid butterfly
starfish, sea star
sea urchin
sea cucumber, holothurian
wood rabbit, cottontail, cottontail rabbit
hare
Angora, Angora rabbit
hamster
porcupine, hedgehog
fox squirrel, eastern fox squirrel, Sciurus niger
marmot
beaver
guinea pig, Cavia cobaya
sorrel
zebra
hog, pig, grunter, squealer, Sus scrofa
wild boar, boar, Sus scrofa
warthog
hippopotamus, hippo, river horse, Hippopotamus amphibius
ox
water buffalo, water ox, Asiatic buffalo, Bubalus bubalis
bison
ram, tup
bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis
ibex, Capra ibex
hartebeest
impala, Aepyceros melampus
gazelle
Arabian camel, dromedary, Camelus dromedarius
llama
weasel
mink
polecat, fitch, foulmart, foumart, Mustela putorius
black-footed ferret, ferret, Mustela nigripes
otter
skunk, polecat, wood pussy
badger
armadillo
three-toed sloth, ai, Bradypus tridactylus
orangutan, orang, orangutang, Pongo pygmaeus
gorilla, Gorilla gorilla
chimpanzee, chimp, Pan troglodytes
gibbon, Hylobates lar
siamang, Hylobates syndactylus, Symphalangus syndactylus
guenon, guenon monkey
patas, hussar monkey, Erythrocebus patas
baboon
macaque
langur
colobus, colobus monkey
proboscis monkey, Nasalis larvatus
marmoset
capuchin, ringtail, Cebus capucinus
howler monkey, howler
titi, titi monkey
spider monkey, Ateles geoffroyi
squirrel monkey, Saimiri sciureus
Madagascar cat, ring-tailed lemur, Lemur catta
indri, indris, Indri indri, Indri brevicaudatus
Indian elephant, Elephas maximus
African elephant, Loxodonta africana
lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens
giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca
barracouta, snoek
eel
coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
rock beauty, Holocanthus tricolor
anemone fish
sturgeon
gar, garfish, garpike, billfish, Lepisosteus osseus
lionfish
puffer, pufferfish, blowfish, globefish
abacus
abaya
academic gown, academic robe, judge's robe
accordion, piano accordion, squeeze box
acoustic guitar
aircraft carrier, carrier, flattop, attack aircraft carrier
airliner
airship, dirigible
altar
ambulance
amphibian, amphibious vehicle
analog clock
apiary, bee house
apron
ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin
assault rifle, assault gun
backpack, back pack, knapsack, packsack, rucksack, haversack
bakery, bakeshop, bakehouse
balance beam, beam
balloon
ballpoint, ballpoint pen, ballpen, Biro
Band Aid
banjo
bannister, banister, balustrade, balusters, handrail
barbell
barber chair
barbershop
barn
barometer
barrel, cask
barrow, garden cart, lawn cart, wheelbarrow
baseball
basketball
bassinet
bassoon
bathing cap, swimming cap
bath towel
bathtub, bathing tub, bath, tub
beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
beacon, lighthouse, beacon light, pharos
beaker
bearskin, busby, shako
beer bottle
beer glass
bell cote, bell cot
bib
bicycle-built-for-two, tandem bicycle, tandem
bikini, two-piece
binder, ring-binder
binoculars, field glasses, opera glasses
birdhouse
boathouse
bobsled, bobsleigh, bob
bolo tie, bolo, bola tie, bola
bonnet, poke bonnet
bookcase
bookshop, bookstore, bookstall
bottlecap
bow
bow tie, bow-tie, bowtie
brass, memorial tablet, plaque
brassiere, bra, bandeau
breakwater, groin, groyne, mole, bulwark, seawall, jetty
breastplate, aegis, egis
broom
bucket, pail
buckle
bulletproof vest
bullet train, bullet
butcher shop, meat market
cab, hack, taxi, taxicab
caldron, cauldron
candle, taper, wax light
cannon
canoe
can opener, tin opener
cardigan
car mirror
carousel, carrousel, merry-go-round, roundabout, whirligig
carpenter's kit, tool kit
carton
car wheel
cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM
cassette
cassette player
castle
catamaran
CD player
cello, violoncello
cellular telephone, cellular phone, cellphone, cell, mobile phone
chain
chainlink fence
chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour
chain saw, chainsaw
chest
chiffonier, commode
chime, bell, gong
china cabinet, china closet
Christmas stocking
church, church building
cinema, movie theater, movie theatre, movie house, picture palace
cleaver, meat cleaver, chopper
cliff dwelling
cloak
clog, geta, patten, sabot
cocktail shaker
coffee mug
coffeepot
coil, spiral, volute, whorl, helix
combination lock
computer keyboard, keypad
confectionery, confectionary, candy store
container ship, containership, container vessel
convertible
corkscrew, bottle screw
cornet, horn, trumpet, trump
cowboy boot
cowboy hat, ten-gallon hat
cradle
crane
crash helmet
crate
crib, cot
Crock Pot
croquet ball
crutch
cuirass
dam, dike, dyke
desk
desktop computer
dial telephone, dial phone
diaper, nappy, napkin
digital clock
digital watch
dining table, board
dishrag, dishcloth
dishwasher, dish washer, dishwashing machine
disk brake, disc brake
dock, dockage, docking facility
dogsled, dog sled, dog sleigh
dome
doormat, welcome mat
drilling platform, offshore rig
drum, membranophone, tympan
drumstick
dumbbell
Dutch oven
electric fan, blower
electric guitar
electric locomotive
entertainment center
envelope
espresso maker
face powder
feather boa, boa
file, file cabinet, filing cabinet
fireboat
fire engine, fire truck
fire screen, fireguard
flagpole, flagstaff
flute, transverse flute
folding chair
football helmet
forklift
fountain
fountain pen
four-poster
freight car
French horn, horn
frying pan, frypan, skillet
fur coat
garbage truck, dustcart
gasmask, respirator, gas helmet
gas pump, gasoline pump, petrol pump, island dispenser
goblet
go-kart
golf ball
golfcart, golf cart
gondola
gong, tam-tam
gown
grand piano, grand
greenhouse, nursery, glasshouse
grille, radiator grille
grocery store, grocery, food market, market
guillotine
hair slide
hair spray
half track
hammer
hamper
hand blower, blow dryer, blow drier, hair dryer, hair drier
hand-held computer, hand-held microcomputer
handkerchief, hankie, hanky, hankey
hard disc, hard disk, fixed disk
harmonica, mouth organ, harp, mouth harp
harp
harvester, reaper
hatchet
holster
home theater, home theatre
honeycomb
hook, claw
hoopskirt, crinoline
horizontal bar, high bar
horse cart, horse-cart
hourglass
iPod
iron, smoothing iron
jack-o'-lantern
jean, blue jean, denim
jeep, landrover
jersey, T-shirt, tee shirt
jigsaw puzzle
jinrikisha, ricksha, rickshaw
joystick
kimono
knee pad
knot
lab coat, laboratory coat
ladle
lampshade, lamp shade
laptop, laptop computer
lawn mower, mower
lens cap, lens cover
letter opener, paper knife, paperknife
library
lifeboat
lighter, light, igniter, ignitor
limousine, limo
liner, ocean liner
lipstick, lip rouge
Loafer
lotion
loudspeaker, speaker, speaker unit, loudspeaker system, speaker system
loupe, jeweler's loupe
lumbermill, sawmill
magnetic compass
mailbag, postbag
mailbox, letter box
maillot
maillot, tank suit
manhole cover
maraca
marimba, xylophone
mask
matchstick
maypole
maze, labyrinth
measuring cup
medicine chest, medicine cabinet
megalith, megalithic structure
microphone, mike
microwave, microwave oven
military uniform
milk can
minibus
miniskirt, mini
minivan
missile
mitten
mixing bowl
mobile home, manufactured home
Model T
modem
monastery
monitor
moped
mortar
mortarboard
mosque
mosquito net
motor scooter, scooter
mountain bike, all-terrain bike, off-roader
mountain tent
mouse, computer mouse
mousetrap
moving van
muzzle
nail
neck brace
necklace
nipple
notebook, notebook computer
obelisk
oboe, hautboy, hautbois
ocarina, sweet potato
odometer, hodometer, mileometer, milometer
oil filter
organ, pipe organ
oscilloscope, scope, cathode-ray oscilloscope, CRO
overskirt
oxcart
oxygen mask
packet
paddle, boat paddle
paddlewheel, paddle wheel
padlock
paintbrush
pajama, pyjama, pj's, jammies
palace
panpipe, pandean pipe, syrinx
paper towel
parachute, chute
parallel bars, bars
park bench
parking meter
passenger car, coach, carriage
patio, terrace
pay-phone, pay-station
pedestal, plinth, footstall
pencil box, pencil case
pencil sharpener
perfume, essence
Petri dish
photocopier
pick, plectrum, plectron
pickelhaube
picket fence, paling
pickup, pickup truck
pier
piggy bank, penny bank
pill bottle
pillow
ping-pong ball
pinwheel
pirate, pirate ship
pitcher, ewer
plane, carpenter's plane, woodworking plane
planetarium
plastic bag
plate rack
plow, plough
plunger, plumber's helper
Polaroid camera, Polaroid Land camera
pole
police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria
poncho
pool table, billiard table, snooker table
pop bottle, soda bottle
pot, flowerpot
potter's wheel
power drill
prayer rug, prayer mat
printer
prison, prison house
projectile, missile
projector
puck, hockey puck
punching bag, punch bag, punching ball, punchball
purse
quill, quill pen
quilt, comforter, comfort, puff
racer, race car, racing car
racket, racquet
radiator
radio, wireless
radio telescope, radio reflector
rain barrel
recreational vehicle, RV, R.V.
reel
reflex camera
refrigerator, icebox
remote control, remote
restaurant, eating house, eating place, eatery
revolver, six-gun, six-shooter
rifle
rocking chair, rocker
rotisserie
rubber eraser, rubber, pencil eraser
rugby ball
rule, ruler
running shoe
safe
safety pin
saltshaker, salt shaker
sandal
sarong
sax, saxophone
scabbard
scale, weighing machine
school bus
schooner
scoreboard
screen, CRT screen
screw
screwdriver
seat belt, seatbelt
sewing machine
shield, buckler
shoe shop, shoe-shop, shoe store
shoji
shopping basket
shopping cart
shovel
shower cap
shower curtain
ski
ski mask
sleeping bag
slide rule, slipstick
sliding door
slot, one-armed bandit
snorkel
snowmobile
snowplow, snowplough
soap dispenser
soccer ball
sock
solar dish, solar collector, solar furnace
sombrero
soup bowl
space bar
space heater
space shuttle
spatula
speedboat
spider web, spider's web
spindle
sports car, sport car
spotlight, spot
stage
steam locomotive
steel arch bridge
steel drum
stethoscope
stole
stone wall
stopwatch, stop watch
stove
strainer
streetcar, tram, tramcar, trolley, trolley car
stretcher
studio couch, day bed
stupa, tope
submarine, pigboat, sub, U-boat
suit, suit of clothes
sundial
sunglass
sunglasses, dark glasses, shades
sunscreen, sunblock, sun blocker
suspension bridge
swab, swob, mop
sweatshirt
swimming trunks, bathing trunks
swing
switch, electric switch, electrical switch
syringe
table lamp
tank, army tank, armored combat vehicle, armoured combat vehicle
tape player
teapot
teddy, teddy bear
television, television system
tennis ball
thatch, thatched roof
theater curtain, theatre curtain
thimble
thresher, thrasher, threshing machine
throne
tile roof
toaster
tobacco shop, tobacconist shop, tobacconist
toilet seat
torch
totem pole
tow truck, tow car, wrecker
toyshop
tractor
trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi
tray
trench coat
tricycle, trike, velocipede
trimaran
tripod
triumphal arch
trolleybus, trolley coach, trackless trolley
trombone
tub, vat
turnstile
typewriter keyboard
umbrella
unicycle, monocycle
upright, upright piano
vacuum, vacuum cleaner
vase
vault
velvet
vending machine
vestment
viaduct
violin, fiddle
volleyball
waffle iron
wall clock
wallet, billfold, notecase, pocketbook
wardrobe, closet, press
warplane, military plane
washbasin, handbasin, washbowl, lavabo, wash-hand basin
washer, automatic washer, washing machine
water bottle
water jug
water tower
whiskey jug
whistle
wig
window screen
window shade
Windsor tie
wine bottle
wing
wok
wooden spoon
wool, woolen, woollen
worm fence, snake fence, snake-rail fence, Virginia fence
wreck
yawl
yurt
web site, website, internet site, site
comic book
crossword puzzle, crossword
street sign
traffic light, traffic signal, stoplight
book jacket, dust cover, dust jacket, dust wrapper
menu
plate
guacamole
consomme
hot pot, hotpot
trifle
ice cream, icecream
ice lolly, lolly, lollipop, popsicle
French loaf
bagel, beigel
pretzel
cheeseburger
hotdog, hot dog, red hot
mashed potato
head cabbage
broccoli
cauliflower
zucchini, courgette
spaghetti squash
acorn squash
butternut squash
cucumber, cuke
artichoke, globe artichoke
bell pepper
cardoon
mushroom
Granny Smith
strawberry
orange
lemon
fig
pineapple, ananas
banana
jackfruit, jak, jack
custard apple
pomegranate
hay
carbonara
chocolate sauce, chocolate syrup
dough
meat loaf, meatloaf
pizza, pizza pie
potpie
burrito
red wine
espresso
cup
eggnog
alp
bubble
cliff, drop, drop-off
coral reef
geyser
lakeside, lakeshore
promontory, headland, head, foreland
sandbar, sand bar
seashore, coast, seacoast, sea-coast
valley, vale
volcano
ballplayer, baseball player
groom, bridegroom
scuba diver
rapeseed
daisy
yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum
corn
acorn
hip, rose hip, rosehip
buckeye, horse chestnut, conker
coral fungus
agaric
gyromitra
stinkhorn, carrion fungus
earthstar
hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa
bolete
ear, spike, capitulum
toilet tissue, toilet paper, bathroom tissue'''.split("\n")