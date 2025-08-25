import math, argparse, os
from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def to_tensor(img: Image.Image):
    '''converts a PIL image to a float tensor in [0,1] with shape [H, W, 3] (RBG)'''
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)

def make_grid(H, W, device):
    '''creates a dense coordinate grid normalized to [-1, 1] in both x and y, then flattens it to shape [H*W, 2] as (x, y) pairs'''
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
    return coords

def psnr(pred: torch.Tensor, target: torch.Tensor):
    '''computes the Peak Signal to Noise Ratio in decibels assuming both pred and target are in [0, 1]'''
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / mse)

def save_image(tensor_hw3, path):
    '''saves a float tensor in [0,1] with shape [H, W, 3] (RBG) as a PNG image'''
    arr = (tensor_hw3.detach().clamp(0,1).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30.0, is_first=False):
        super().__init__()
        self.in_f, self.out_f, self.w0 = in_f, out_f, w0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_f, 1 / self.in_f)
            else:
                bound = math.sqrt(6 / self.in_f) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.fill_(0.0)
        
    def forward(self , x):
        return torch.sin(self.w0 * self.linear(x))

class SirenMLP(nn.Module):
    def __init__(self, in_f=2, hidden=256, depth=5, out_f=3, w0=30.0):
        super().__init__()
        layers = []
        layers.append(SirenLayer(in_f, hidden, w0=w0, is_first=True))
        for _ in range(depth - 2):
            layers.append(SirenLayer(hidden, hidden, w0=1.0, is_first=False))
        self.hidden = nn.Sequential(*layers)
        self.final = nn.Linear(hidden, out_f)
        with torch.no_grad():
            self.final.weight.uniform_(-1e-4, 1e-4)
            self.final.bias.fill_(0.0)

    def forward(self, xy):
        h = self.hidden(xy)
        rgb = torch.sigmoid(self.final(h))  
        return rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, default="image.png")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--w0", type=float, default=30.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="runs/siren")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(args.image_path)
    H, W = img.height, img.width
    target = to_tensor(img).to(device).view(-1, 3)  

    coords = make_grid(H, W, device)  

    model = SirenMLP(in_f=2, hidden=args.hidden, depth=args.depth, out_f=3, w0=args.w0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        pred = model(coords) 
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                _psnr = psnr(pred, target).item()
            print(f"step {step:05d} | loss {loss.item():.6f} | PSNR {_psnr:.2f}dB")

        if step % args.save_every == 0 or step == args.steps:
            with torch.no_grad():
                recon = pred.view(H, W, 3)
            save_image(recon, os.path.join(args.out_dir, f"recon_{step:05d}.png"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "siren.pth"))
    print("done.")

if __name__ == "__main__":
    main()