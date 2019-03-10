import math
import cv2
import numpy
from keras.models import Model
from keras.layers.convolutional import  Conv2D,UpSampling2D, Conv2DTranspose,Cropping2D
#--------------------------------------------------------------------------------------------------------------------------
import tools_image

# ----------------------------------------------------------------------------------------------------------------------
def inverce_weight(W,b_size):

    F = W[0]

    if len(W[1])==b_size:
        B = W[1].copy()
    else:
        B = numpy.full(b_size,W[1].mean(),numpy.float32)
    return [F,B]
# ----------------------------------------------------------------------------------------------------------------------
def colorize_chess(image_gray,W,H):

    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    r, c = numpy.meshgrid(numpy.linspace(0, H - 1, num=H), numpy.linspace(0, W - 1, num=W))
    r, c = r.astype(numpy.int), c.astype(numpy.int)


    for R in numpy.arange(0, image_gray.shape[0], H):
        for C in numpy.arange(0, image_gray.shape[1], W):
            if ((int(R/H)+int(C/W))%2 == 0):
                red, green, blue = 0.95, 0.95, 1.05
            else:
                red, green, blue = 1.05, 1.05, 0.95
            image_color[R + r, C + c, 0] = numpy.clip(image_color[R + r, C + c, 0] * blue, 0, 255)
            image_color[R + r, C + c, 1] = numpy.clip(image_color[R + r, C + c, 1] * green, 0, 255)
            image_color[R + r, C + c, 2] = numpy.clip(image_color[R + r, C + c, 2] * red, 0, 255)

    return image_color
# ----------------------------------------------------------------------------------------------------------------------
def tensor_gray_1D_to_image(tensor,orientation = 'landscape'):

    rows = tools_image.numerical_devisor(tensor.shape[0])
    cols = int(tensor.shape[0] / rows)

    if orientation=='landscape':
        image = numpy.reshape(tensor,(numpy.minimum(rows,cols),numpy.maximum(rows, cols)))
    else:
        image = numpy.reshape(tensor, (numpy.maximum(rows, cols), numpy.minimum(rows, cols)))


    return image
# ----------------------------------------------------------------------------------------------------------------------
def tensor_gray_3D_to_image(tensor, do_colorize = False):


    rows = tools_image.numerical_devisor(tensor.shape[2])

    h, w, R, C = tensor.shape[0], tensor.shape[1], rows, int(tensor.shape[2] / rows)
    image = numpy.zeros((h * R, w * C), dtype=numpy.float32)
    for i in range(0, tensor.shape[2]):
        col, row = i % C, int(i / C)
        image[h * row:h * row + h, w * col:w * col + w] = tensor[:, :, i]

    if do_colorize:
        image = colorize_chess(image,tensor.shape[0],tensor.shape[1])

    return image
# ---------------------------------------------------------------------------------------------------------------------
def image_to_tensor_color_4D(image,shape):
    tensor = numpy.zeros(shape,numpy.float32)

    h,w =shape[0],shape[1]
    rows = tools_image.numerical_devisor(shape[3])
    C = int(tensor.shape[3] / rows)

    for i in range(0,96):
        col, row = i % C, int(i / C)
        tensor[:, :, :, i]=image[h * row:h * row + h, w * col:w * col + w]

    return tensor
# ---------------------------------------------------------------------------------------------------------------------
def tensor_color_4D_to_image(tensor):


    rows = tools_image.numerical_devisor(tensor.shape[3])

    h, w, R, C = tensor.shape[0], tensor.shape[1], rows, int(tensor.shape[3] / rows)
    image = numpy.zeros((h * R,w * C, tensor.shape[2]),dtype=numpy.float32)
    for i in range(0, tensor.shape[3]):
        col, row = i % C, int(i / C)
        image[h * row:h * row + h, w * col:w * col + w, :] = tensor[:, :, :, i]


    return image
# ---------------------------------------------------------------------------------------------------------------------
def tensor_gray_4D_to_image(tensor,do_colorize = False):

    sub_image = tensor_gray_3D_to_image(tensor[:, :, :, 0])
    h, w = sub_image.shape[0], sub_image.shape[1]


    R = tools_image.numerical_devisor(tensor.shape[3])
    C = int(tensor.shape[3]/R)


    if do_colorize:
        image = numpy.zeros((h * R, w * C, 3), dtype=numpy.float32)
    else:
        image = numpy.zeros((h * R, w * C), dtype=numpy.float32)
    for i in range (0,tensor.shape[3]):
        col, row = i % C, int(i / C)
        sub_image = tensor_gray_3D_to_image(tensor[:,:,:,i],do_colorize)
        if do_colorize:
            image[h * row:h * row + h, w * col:w * col + w,:] = sub_image[:, :, :]
        else:
            image[h * row:h * row + h, w * col:w * col + w] = sub_image[:, :]

    return image
# ---------------------------------------------------------------------------------------------------------------------
def visualize_filters(keras_model, path_out):

    for i in range(0, len(keras_model.layers)):
        tensor = keras_model.layers[i]
        if isinstance(tensor, Conv2D):
            tensor = tensor.get_weights()[0]
            tensor -= tensor.min()
            tensor *= 255.0 / tensor.max()  # (W,H,3,N)

            if tensor.shape[2] == 3:
                cv2.imwrite(path_out + 'filter_%03d.png' % i, tensor_color_4D_to_image(tensor))
            else:
                cv2.imwrite(path_out + 'filter_%03d.png' % i, tensor_gray_4D_to_image(tensor, do_colorize=True))

    return
# ---------------------------------------------------------------------------------------------------------------------
def stage_tensors(outputs,path_out):

    for i in range(0, len(outputs)):
        tensor = outputs[i][0]
        tensor -= tensor.min()
        tensor *= 255.0 / tensor.max()

        if len(tensor.shape) == 3:
            if tensor.shape[2] == 3:  cv2.imwrite(path_out + 'layer_%03d.png' % i, tensor)
            elif tensor.shape[2] != 3:cv2.imwrite(path_out + 'layer_%03d.png' % i, tensor_gray_3D_to_image(tensor, do_colorize=True))
        elif len(tensor.shape) == 1:  cv2.imwrite(path_out + 'layer_%03d.png' % i, tools_image.hitmap2d_to_viridis(tensor_gray_1D_to_image(tensor)))

    return
# ---------------------------------------------------------------------------------------------------------------------
def visualize_layers(keras_model, filename_input, path_out,need_scale=False):
    image = cv2.imread(filename_input)

    shape = keras_model.input.get_shape().as_list()
    need_transpose = True if shape[3]!=3 else False

    if need_transpose:
        H,W = shape[2], shape[3]
        if H is None: H=64
        if W is None: W=64
        image = cv2.resize(image, (H,W))
        image = image.transpose((2, 0, 1))
    else:
        H,W = shape[1], shape[2]
        if H is None: H=64
        if W is None: W=64
        image = cv2.resize(image, (H,W))

    if need_scale==True:
        image=normalize(image.astype(numpy.float32))

    outputs = Model(inputs=keras_model.input, outputs=[layer.output for layer in keras_model.layers]).predict(numpy.array([image]))

    #if need_scale==True:
    #    outputs = [scale(each) for each in outputs]

    stage_tensors(outputs,path_out)


    return
# ---------------------------------------------------------------------------------------------------------------------
def add_de_pool_layer(invmodel, orig_layer_after,input_shape=None):
    scale = orig_layer_after.input.get_shape().as_list()[1]/orig_layer_after.output.get_shape().as_list()[1]
    scale = int(scale+0.5)

    if input_shape is not None:
        invmodel.add(UpSampling2D(size=(scale, scale), input_shape=input_shape))
    else:
        invmodel.add(UpSampling2D(size=(scale, scale)))
    return invmodel
# ---------------------------------------------------------------------------------------------------------------------
def add_de_conv_layer(invmodel,orig_conv_layer,crop=None,input_shape=None):
    N = int(orig_conv_layer.input.get_shape()[3])

    padding=orig_conv_layer.padding


    if input_shape is None:
        invmodel.add(Conv2DTranspose(N, orig_conv_layer.kernel_size,activation=orig_conv_layer.activation,
                                     padding=padding,data_format='channels_last'))
    else:
        invmodel.add(Conv2DTranspose(N, orig_conv_layer.kernel_size,activation=orig_conv_layer.activation,
                                     padding=padding,data_format='channels_last',
                                     input_shape=input_shape))

    w = orig_conv_layer.get_weights()
    b = orig_conv_layer.input.get_shape()[3]
    iw = inverce_weight(w,b)
    invmodel.layers[-1].set_weights(iw)

    invmodel.add(UpSampling2D(size=orig_conv_layer.strides))

    if crop is not None:
        invmodel.add(Cropping2D( ((0, crop), (0, crop)) ))
    return invmodel
# ---------------------------------------------------------------------------------------------------------------------
def import_weight(filename,shape):
    image = cv2.imread(filename)
    weights = image_to_tensor_color_4D(image,shape).astype(numpy.float32)
    #cv2.imwrite('data/output/filter01.png',tensor_color_4D_to_image(filters))
    weights /= 255.0
    weights -= 0.5
    weights *= 2

    return weights
# ---------------------------------------------------------------------------------------------------------------------
def import_bias(n=96):
    if n==96:
        B = numpy.array([-0.3463971,0.2838365,-0.49968186,-0.23649067,-0.5688867,-0.6436641,-0.6924773,-0.4955547,-0.23227385,0.25512516,-0.26485363,-0.41346544,-0.43461457,-0.70009106,-0.47496662,-0.47548744,-0.69679964,-0.46574312,-0.52741444,-0.7435926,-0.6915156,-0.50525445,-0.2625096,-0.65133756,-0.4111215,-0.5997434,-0.68655,-0.5016767,-0.72232133,-0.60682845,-0.6017115,-0.49666974,-0.5611206,-0.48387265,-0.6110909,-0.54242045,-0.56991154,-0.43601885,-0.3956422,-0.37117276,0.036626954,-0.22927734,0.0961724,-0.480799,0.020177085,-0.7724009,-0.63782954,-0.5926624,-0.4724126,-0.5151979,-0.7155822,-0.2635522,-0.62939286,-0.3237968,-0.6559588,-0.08852144,-0.63524127,-0.58848786,-0.5521372,-0.32681417,-0.48454055,-0.6135006,-0.6231938,-0.7153859,-0.7460073,-0.5911307,-0.38240275,-0.31587332,-0.6460482,-0.5697083,-0.675176,-0.708902,-0.410006,-0.5047665,-0.5587012,-0.6281784,-0.086676925,-0.5107954,0.17226526,-0.54705405,-0.5593643,-0.55164367,-0.18164437,-0.014679801,-0.2994107,-0.38288164,-0.0946849,-0.09597142,-0.26413,-0.46921661,-0.45871097,-0.6019412,-0.33356422,-0.5627348,-0.39224792,-0.8009203])
    else:
        B = numpy.zeros(n,numpy.float32)
    return B
# ---------------------------------------------------------------------------------------------------------------------
def construct_filters_2x2x3():

    return [construct_weights_2x2x3(),numpy.zeros(48,numpy.float32)]
# ---------------------------------------------------------------------------------------------------------------------
def construct_weights_2x2x3(n=16*3):
    F = numpy.zeros((2, 2, 3, 16*3))
    for c,i in zip([0,1,0],[0,16,16*2]):
        F[:, :, c, i+ 0] = numpy.array([[-1, -1], [-1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 1] = numpy.array([[-1, -1], [-1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 2] = numpy.array([[-1, -1], [+1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 3] = numpy.array([[-1, -1], [+1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 4] = numpy.array([[-1, +1], [-1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 5] = numpy.array([[-1, +1], [-1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 6] = numpy.array([[-1, +1], [+1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 7] = numpy.array([[-1, +1], [+1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 8] = numpy.array([[+1, -1], [-1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+ 9] = numpy.array([[+1, -1], [-1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+10] = numpy.array([[+1, -1], [+1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+11] = numpy.array([[+1, -1], [+1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+12] = numpy.array([[+1, +1], [-1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+13] = numpy.array([[+1, +1], [-1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+14] = numpy.array([[+1, +1], [+1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, i+15] = numpy.array([[+1, +1], [+1, +1]]).astype(numpy.float32)/4.0

    return F / 3.0
# ---------------------------------------------------------------------------------------------------------------------
def construct_weights_2x2(n=16):

    F = numpy.zeros((2, 2, 3,16))

    for c in range (0,3):
        F[:, :, c,  0] = numpy.array([[-1, -1], [-1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c,  1] = numpy.array([[-1, -1], [-1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c,  2] = numpy.array([[-1, -1], [+1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c,  3] = numpy.array([[-1, -1], [+1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c,  4] = numpy.array([[-1, +1], [-1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c,  5] = numpy.array([[-1, +1], [-1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c,  6] = numpy.array([[-1, +1], [+1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c,  7] = numpy.array([[-1, +1], [+1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c,  8] = numpy.array([[+1, -1], [-1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c,  9] = numpy.array([[+1, -1], [-1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, 10] = numpy.array([[+1, -1], [+1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, 11] = numpy.array([[+1, -1], [+1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, 12] = numpy.array([[+1, +1], [-1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, 13] = numpy.array([[+1, +1], [-1, +1]]).astype(numpy.float32)/4.0
        F[:, :, c, 14] = numpy.array([[+1, +1], [+1, -1]]).astype(numpy.float32)/4.0
        F[:, :, c, 15] = numpy.array([[+1, +1], [+1, +1]]).astype(numpy.float32)/4.0

    return F/3.0
# ---------------------------------------------------------------------------------------------------------------------
def construct_filters_2x2(n_filters=16):

    return [construct_weights_2x2()[:,:,:,:n_filters],numpy.zeros(16,numpy.float32)[:n_filters]]
# ----------------------------------------------------------------------------------------------------------------------
def normalize(y):
    x=y.copy()
    x/= 255.0
    x-= 0.5
    x*= 2.0
    return x
# ----------------------------------------------------------------------------------------------------------------------
def scale(y):
    x = y.copy()
    x/= 2.0
    x += 0.5
    x *= 255.0
    return x
# ----------------------------------------------------------------------------------------------------------------------

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