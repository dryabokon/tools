from torchvision.utils import save_image, make_grid
from torch.utils import data
from torchvision import transforms as T
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy
import os
import cv2
from PIL import Image
import tools_IO
import tools_image
#----------------------------------------------------------------------------------------------------------------------
class CelebA(data.Dataset):
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):

        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def preprocess(self):
        with open(self.attr_path) as f:
            raw_lines = f.readlines()

        lines = []
        for line in raw_lines:lines.append(line.rstrip())
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        return
# ----------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)
# ----------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return self.num_images
#----------------------------------------------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

#----------------------------------------------------------------------------------------------------------------------
class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)
#----------------------------------------------------------------------------------------------------------------------
class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / numpy.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
#----------------------------------------------------------------------------------------------------------------------
class Face_filter(object):
    def __init__(self, result_dir,filename_G_weights=None,filename_D_weights=None):


        # Model configurations.
        self.c_dim = 5
        self.c2_dim = 8
        self.image_size = 128
        self.g_conv_dim = 64
        self.d_conv_dim = 64
        self.g_repeat_num = 6
        self.d_repeat_num = 6
        self.lambda_cls = float(1.0)
        self.lambda_rec = float(10.0)
        self.lambda_gp = float(10.0)

        # Training configurations.
        #self.dataset = config.dataset
        self.batch_size = 32
        self.num_iters = 2000
        self.num_iters_decay = 1000
        self.g_lr = float(0.0001)
        self.d_lr = float(0.0001)
        self.n_critic = 5
        self.beta1 = float(0.5)
        self.beta2 = float(0.999)
        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

        # Test configurations.
        self.test_iters = 2000

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.result_dir = result_dir
        self.log_dir = result_dir
        self.sample_dir = result_dir
        self.model_save_dir = result_dir


        # Step size.
        self.log_step = 10
        self.sample_step = 100
        self.model_save_step = 10000
        self.lr_update_step = 1000

        self.build_model()
        if filename_G_weights is not None and filename_D_weights is not None:
            self.restore_model(filename_G_weights,filename_D_weights)

        self.transform = []
        self.transform.append(T.CenterCrop(178))
        self.transform.append(T.Resize(self.image_size))
        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T.Compose(self.transform)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def build_model(self):

        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        self.G.to(self.device)
        self.D.to(self.device)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def restore_model(self,filename_G_weights,filename_D_weights):
        self.G.load_state_dict(torch.load(filename_G_weights, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(filename_D_weights, map_location=lambda storage, loc: storage))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        return
# ----------------------------------------------------------------------------------------------------------------------
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
# ----------------------------------------------------------------------------------------------------------------------
    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,inputs=x,grad_outputs=weight,retain_graph=True,create_graph=True,only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
# ----------------------------------------------------------------------------------------------------------------------
    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[numpy.arange(batch_size), labels.long()] = 1
        return out
# ----------------------------------------------------------------------------------------------------------------------
    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list
# ----------------------------------------------------------------------------------------------------------------------
    def classification_loss(self, logit, target, dataset='CelebA'):
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)
# ----------------------------------------------------------------------------------------------------------------------
    def test(self):

        with torch.no_grad():

            for i, (x_real, c_org) in enumerate(self.data_loader):

                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, c_dim=self.c_dim, selected_attrs=self.selected_attrs)

                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_file(self, filename_in,filename_out):

        image = Image.open(filename_in)
        image_original = numpy.array(image)
        image_original = tools_image.center_crop(image_original, self.image_size * 2)

        x_real = self.transform(image)
        x_real = x_real.unsqueeze(0)
        image_result = numpy.zeros((self.image_size*2,self.image_size*4,3),dtype=numpy.uint8)
        image_result = tools_image.put_image(image_result, image_original, 0, 0)

        #Black_Hair, Blond_Hair, Brown_Hair, Male, 'Young
        #c_org = torch.tensor([[0.0,0.0,1.0,0.0,1.0]])
        c_org = torch.tensor([[0.0,0.0,1.0,1.0,1.0]])
        c_trg_list = self.create_labels(c_org, c_dim=self.c_dim, selected_attrs=self.selected_attrs)

        for i,c_trg in enumerate(c_trg_list):
            res = self.G(x_real, c_trg)
            tensor = self.denorm(res.data.cpu())
            grid_result = make_grid(tensor, nrow=1, padding=0)
            newlook = grid_result.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

            if i == 0: image_result=tools_image.put_image(image_result, newlook, 0,self.image_size*2+ 0)
            if i == 1: image_result=tools_image.put_image(image_result, newlook, 0,self.image_size*2+ self.image_size)
            if i == 2: image_result=tools_image.put_image(image_result, newlook, self.image_size, self.image_size*2+0)
            if i == 4: image_result=tools_image.put_image(image_result, newlook, self.image_size, self.image_size*2+self.image_size)
        image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename_out,image_result)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder(self, path_input, path_out, list_of_masks='*.png,*.jpg', limit=1000000):
        tools_IO.remove_files(path_out,create=True)
        start_time = time.time()
        local_filenames  = tools_IO.get_filenames(path_input, list_of_masks)[:limit]
        local_filenames = numpy.sort(local_filenames)
        for local_filename in local_filenames:
            self.process_file(path_input + local_filename,path_out + local_filename)

        total_time = (time.time() - start_time)
        print('Processing: %s sec in total - %f per image' % (total_time, int(total_time) / len(local_filenames)))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def train(self,image_dir, attr_path):

        dataset = CelebA(image_dir, attr_path, self.selected_attrs, self.transform, 'test')
        self.data_loader = data.DataLoader(dataset=dataset, batch_size=1, num_workers=1)

        # Fetch fixed inputs for debugging.
        data_iter = iter(self.data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, c_dim=self.c_dim, selected_attrs=self.selected_attrs)


        for i in range(0, self.num_iters):
            print('.',end='')

            #1.Preprocess input data
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()


            x_real = x_real.to(self.device)  # Input images.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(self.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)  # Labels for computing classification loss.


            #2.Train the discriminator

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            #3. Train the generator                                #
            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()



            #4. Miscellaneous                                    #
            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))


        return
# ---------------------------------------------------------------------------------------------------------------------