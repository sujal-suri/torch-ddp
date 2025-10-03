import torch
import torchvision
import torch.nn as nn
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    num_epochs: int = 50
    wd: float= 1e-5
    outputs: str = "./checkpoints"
    device = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(cfg, model, train_dataloader, optimizer, loss_fn):
    model.train()
    epoch_loss = 0.0

    for _, batch in enumerate(tqdm(train_dataloader)):
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
        for _, batch in enumerate(tqdm(valid_dataloader)):
            input, labels = batch
            input, labels = input.to(cfg.device), labels.to(device)
            outputs = model(input)
            loss = loss_fn(outputs, labels)
            # \sum_{i=1}^{n+1} = \frac{\sum_{i=1}^{n} \cdot (n) + a_{n+1}}{n+1}
            valid_loss = (valid_loss * _ + loss)/(_ + 1)

    return epoch_loss        

def plot_history(cfg: Config, train_loss:list , valid_loss: list) -> None:
    plt.plot(x=list(range(len(train_loss))), y=train_loss, color='red', label='Train Loss')
    plt.plot(x=list(range(len(valid_loss))), y=valid_loss, color='blue', label='Valid Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train/Valid Loss Curve')
    plt.legend()

    plt.savefig(f"{cfg.outputs}/loss_curve.png")

def train(cfg: Config, model: nn.Module, train: Dataset, valid: Dataset):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    
    train_dataloader = DataLoader(train, batch_size = cfg.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size = cfg.valid_batch_size, shuffle=True)

    best_val_loss = float('inf')
    training_losses, valid_losses = [], []

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(cfg, model, train_dataloader, optimizer, loss_fn)
        val_loss = valid_epoch(cfg, model, valid_dataloader, loss_fn)
        
        if(val_loss < best_val_loss):
            best_val_loss = val_loss
            print(f"Saving Best Model at {cfg.outputs}/best_model")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,                                    
            }, f"{cfg.outputs}/best_model/checkpoint.pth")
            
        training_losses.append(train_loss)
        valid_losses.append(val_loss)

        print(f"Training epoch: {epoch+1}| train_loss: {train_loss:.3f}, valid_loss: {val_loss:.3f}")

    print("Saving the Final Model at {cfg.outputs}/final_model")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,                                    
    }, f"{cfg.outputs}/final_model/checkpoint.pth")

    plot_history(cfg, training_losses, valid_losses)

    
def main():
    cfg = Config()
    model = torchvision.models.resnet50(weights=None, progress=True)
    model.fc = nn.Linear(2048, 500)

    model.to(cfg.device)
    train_dataset = TinyImageNet("train")
    valid_dataset = TinyImageNet("valid")
    train(cfg, model, train_dataset, valid_dataset) 

if __name__ == "__main__":
    main()
