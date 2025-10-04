import os
import torch
import torchvision
import torch.nn as nn
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# Import for DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def setup_ddp(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

class TinyImageNet(Dataset):
    def __init__(self, split: str="train") -> None:
        super().__init__()
        self.data = load_dataset("zh-plus/tiny-imagenet", split=split)
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        pil_image, label = self.data[idx]['image'].convert('RGB'), self.data[idx]['label'] 
        return (self.transform(pil_image), label)

@dataclass
class Config:
    lr: float = 1e-3
    train_batch_size: int = 64
    valid_batch_size: int = 128
    num_epochs: int = 1
    wd: float= 1e-5
    outputs: str = "./checkpoints"
    gpu_rank:int = -1
    world_size: int = -1

def train_one_epoch(cfg, model, train_dataloader, optimizer, loss_fn):
    model.train()
    epoch_loss = 0.0

    for _, batch in enumerate(train_dataloader):
        input, labels = batch
        input, labels = input.to(cfg.device), labels.to(cfg.device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        #\sum_{i=1}^{n+1} = \frac{\sum_{i=1}^{n} \cdot (n) + a_{n+1}}{n+1}
        epoch_loss = (epoch_loss * _ + loss)/(_ + 1)

    return epoch_loss        

def valid_epoch(cfg, model, valid_dataloader, loss_fn):
    model.eval()

    valid_loss = 0.0
    with torch.inference_mode():
        for _, batch in enumerate(valid_dataloader):
            input, labels = batch
            input, labels = input.to(cfg.device), labels.to(cfg.device)
            outputs = model(input)
            loss = loss_fn(outputs, labels)
            # \sum_{i=1}^{n+1} = \frac{\sum_{i=1}^{n} \cdot (n) + a_{n+1}}{n+1}
            valid_loss = (valid_loss * _ + loss)/(_ + 1)

    return valid_loss        

def plot_history(cfg: Config, train_loss:list , valid_loss: list) -> None:
    plt.plot(list(range(len(train_loss))), train_loss, color='red', label='Train Loss')
    plt.plot(list(range(len(valid_loss))), valid_loss, color='blue', label='Valid Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train/Valid Loss Curve')
    plt.legend()

    plt.savefig(f"{cfg.outputs}/loss_curve.png")

def prepare_dataloader(cfg, dataset, split):
    world_size, gpu_rank = cfg.world_size, cfg.gpu_rank
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=gpu_rank, shuffle=False, drop_last=False)
    bsz = cfg.train_batch_size if split == "train" else cfg.valid_batch_size
    dataloader = DataLoader(dataset, batch_size = split, pin_memory=False, drop_last=False, shuffle=False, sampler=sampler, num_workers=0)
    return dataloader

def train(cfg: Config, model: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    best_val_loss = None
    training_losses, valid_losses = None, None
    if(cfg.rank == 0):
        best_val_loss = float('inf')
        training_losses, valid_losses = [], []

    for epoch in range(cfg.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        valid_dataloader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(cfg, model, train_dataloader, optimizer, loss_fn)
        valid_loss = valid_epoch(cfg, model, valid_dataloader, loss_fn)
             
        train_loss_gather_list = None
        valid_loss_gather_list = None
        if(dist.get_rank == 0):
            train_loss_gather_list = [torch.zeros_like(train_loss) for _ in range(cfg.world_size)]
            valid_loss_gather_list = [torch.zeros_like(valid_loss) for _ in range(cfg.world_size)]

        dist.gather(tensor = train_loss, gather_list = train_loss_gather_list, dst = 0)
        dist.gather(tensor = valid_loss, gather_list = valid_loss_gather_list, dst = 0)

        valid_loss = torch.mean(torch.stack(valid_loss_gather_list))
        train_loss = torch.mean(torch.stack(train_loss_gather_list))
        
        if(cfg.rank == 0):
            if(valid_loss < best_valid_loss):
                print(f"Saving Best Model at {cfg.outputs}/best_model")
                best_valid_loss = valid_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, f"{cfg.outputs}/best_model/checkpoint.pth")
            
            training_losses.append(train_loss.clone().detach().cpu().numpy())
            valid_losses.append(valid_loss.clone().detach().cpu().numpy())

            print(f"Training epoch: {epoch+1}| train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}")

    if(cfg.rank == 0):
        print("Saving the Final Model at {cfg.outputs}/final_model")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"{cfg.outputs}/final_model/checkpoint.pth")

        plot_history(cfg, training_losses, valid_losses)

def cleanup():
    dist.destory_process_group()
    
def main(rank, world_size):
    cfg = Config(gpu_rank=rank, world_size=world_size)
    setup_ddp(rank, world_size)

    os.makedirs(f"{cfg.outputs}/best_model", exist_ok=True)
    os.makedirs(f"{cfg.outputs}/final_model", exist_ok=True)

    model = torchvision.models.resnet50(weights=None, progress=True)
    model.fc = nn.Linear(2048, 500)

    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    train_dataloader = prepare_dataloader(cfg, TinyImageNet("train"), "train") 
    valid_dataloader = prepare_dataloader(cfg, TinyImageNet("valid"), "valid")

    train(cfg, model, train_dataloader, valid_dataloader) 
    cleanup()

if __name__ == "__main__":
    world_size = 2

    mp.spawn(
            main,
            args = (world_size),
            nprocs = world_size
            )
