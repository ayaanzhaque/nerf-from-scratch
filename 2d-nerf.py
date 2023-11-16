import re
import math
import imageio
import os

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import io

# ChatGPT generated
def create_gif(image_folder, gif_path, duration=0.1):
    def sort_key(filename):
        number = re.search(r"(\d+)", filename)
        return int(number.group(1)) if number else 0

    filenames = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('0.png') or f.endswith('0.jpg')],
                       key=sort_key)
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(gif_path, images, duration=duration)


def positional_encoding(x, L):
    freqs = 2.0 ** torch.arange(L).float().to(x.device)
    x_input = x.unsqueeze(-1) * freqs * 2 * torch.pi
    encoding = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
    encoding = torch.cat([x, encoding.reshape(*x.shape[:-1], -1)], dim=-1) # add to original input    
    return encoding

def predict_image(model, image_shape, device):
    h, w = image_shape
    x = torch.linspace(0, w-1, w).repeat(h, 1).to(device) / (w)  # Normalize by dividing by width
    y = torch.linspace(0, h-1, h).repeat(w, 1).transpose(0, 1).to(device) / (h)  # Normalize by dividing by height
    all_coords = torch.stack([x, y], dim=-1).view(-1, 2)

    with torch.no_grad():
        predicted_pixels = model(all_coords)
    
    predicted_image = predicted_pixels.reshape(h, w, 3).cpu().numpy()
    return predicted_image

def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class MLP(nn.Module):
    def __init__(self, L=10):
        super(MLP, self).__init__()
        self.L = L
        
        self.fc1 = nn.Linear(2 * 2 * L + 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = positional_encoding(x, L=self.L)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x

# 4. Define the training loop
def train_model(model, image, optimizer, criterion, iters=3000, batch_size=1000, device='cuda'):
    
    psnr_scores = []
    
    model.train()
    h, w, c = image.shape
    
    x = torch.linspace(0, w-1, w).repeat(h, 1).to(device) / (w)  # Normalize by dividing by width
    y = torch.linspace(0, h-1, h).repeat(w, 1).transpose(0, 1).to(device) / (h)  # Normalize by dividing by height
    all_coords = torch.stack([x, y], dim=-1).view(-1, 2)

    all_pixels = image.view(-1, 3).to(device)
    
    for iteration in range(iters):
        idx = torch.randint(0, h*w, (batch_size,), device=device)
        coords_batch = all_coords.reshape(-1, 2)[idx]
        pixel_batch = all_pixels[idx]
        
        optimizer.zero_grad()
        outputs = model(coords_batch)
        loss = criterion(outputs, pixel_batch)
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration+1}/{iters}, Loss: {loss.item():.6f}")
            curr_psnr = psnr(image.cpu().numpy(), predict_image(model, (h, w), device))
            print(f"PSNR: {curr_psnr:.2f} dB")
            psnr_scores.append(curr_psnr)
            
            model.eval()
            predicted_image = predict_image(model, (h, w), device)
            plt.imsave(f"lion/iter{iteration+1}.jpg", predicted_image)
            model.train()
    
    # create psnr plot
    plt.figure()
    plt.plot(range(100, 3001, 100), psnr_scores)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Iteration')
    plt.savefig('plots/psnr_lion.png')
    
    final_image = predict_image(model, image.shape[:2], device)
    plt.imsave("lion/final.png", final_image)
    
    create_gif('lion', 'lion/training.gif')
    print(f"PSNR: {psnr(image.cpu().numpy(), final_image):.2f} dB")


if __name__ == "__main__":
    device = torch.device('cuda:1')
    image_path = "data/lion.jpeg"
    image = io.read_image(image_path).float() / 255.0
    image = image.permute(1, 2, 0)
    image = image.to(device)

    model = MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    train_model(model, image, optimizer, criterion, iters=3000, batch_size=image.shape[0] * image.shape[1], device=device)

