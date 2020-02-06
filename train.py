import torch
from tqdm import tqdm

from dataset import get_train_data_loader
from model import ResidualBlock, ResNet
from utils.scheduler import GradualWarmupScheduler
from utils.utils import device, logger

# Hyper Parameters
num_epochs = 100
learning_rate = 0.001


def main():
    model = ResNet(ResidualBlock).to(device)
    model.reload()
    model.train()
    logger.info("Train: Init model")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_after = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5
    )
    scheduler = GradualWarmupScheduler(
        optimizer, 8, 10, after_scheduler=scheduler_after
    )

    train_dataloader = get_train_data_loader()
    loss_best = 1
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.long()
            label1, label2 = labels[:, 0], labels[:, 1]

            optimizer.zero_grad()
            y1, y2 = model(images)
            loss1, loss2 = criterion(y1, label1), criterion(y2, label2)
            loss = loss1 + loss2
            # outputs = model(images)
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.info(f"epoch: {epoch}, step: {i}, loss: {loss.item()}")
        model.save()
        if loss_best > loss.item():
            loss_best = loss.item()
            torch.save(model.state_dict(), "model/best.pkl")
            logger.info("Train: Saved best model")
    torch.save(model.state_dict(), "model/final.pkl")
    logger.info("Train: Saved last model")


if __name__ == "__main__":
    main()
