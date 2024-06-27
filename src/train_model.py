import os, argparse, json

import torch
import torch.nn as nn
import torch.utils
import torchvision
import torch.optim as optim

from model import DehazeModel
import dataloader

def weights_init(m):
    """
    Initialize weights
    """
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    """
    Train function
    """
    model = DehazeModel().cuda()
    model.apply(weights_init)

    train_dataset = dataloader.DehazingLoader(config['orig_images_path'], config['hazy_images_path'])
    val_dataset = dataloader.DehazingLoader(config['orig_images_path'], config['hazy_images_path'], mode ='val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['train_batch_size'], shuffle = True, num_workers = config['num_workers'], pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['val_batch_size'], shuffle = True, num_workers = config['num_workers'], pin_memory = True)

    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    model.train()

    for epoch in range(config['num_epochs']):
        for iteration, (img_clear, img_hazy) in enumerate(train_loader):
            img_clear = img_clear.cuda()
            img_hazy = img_hazy.cuda()

            clear_image = model(img_hazy)

            loss = criterion(clear_image, img_clear)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])

            optimizer.step()

            if ((iteration + 1) % config['display_iter']) == 0:
                print(f'Loss at iteration {iteration + 1}: {loss.item()}')
            
            if ((iteration + 1) % config['snapshot_iter']) == 0:
                torch.save(model.state_dict(), os.path.join(config['weights_folder'], f'Epoch {epoch}.pt'))

        for iter_val, (img_clear, img_haze) in enumerate(val_loader):

            img_clear = img_clear.cuda()
            img_haze = img_haze.cuda()

            clear_image = model(img_haze)

            torchvision.utils.save_image(torch.cat((img_haze, clear_image, img_clear), 0), os.path.join(config['examples_folder'], f"{iter_val + 1}.jpg"))

        torch.save(model.state_dict(), os.path.join(config['weights_folder'], "dehazer.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = 'config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if not os.path.exists(config['weights_folder']):
        os.mkdir(config['weights_folder'])
    
    if not os.path.exists(config['examples_folder']):
        os.mkdir(config['examples_folder'])

    train(config)


