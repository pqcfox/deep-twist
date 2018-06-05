import torchvision
from deep_twist.data import utils, dataset, transforms
from skimage import io

train_transform = torchvision.transforms.Compose([transforms.CenterCrop(351),
                                                  transforms.RandomRotate(0, 360),
                                                  transforms.CenterCrop(321),
                                                  transforms.RandomTranslate(50),
                                                  transforms.Resize(227)])

val_transform = torchvision.transforms.Compose(...)


def train_model(train_data, dev_data, model, args):
    train_data.transform = train_transform

    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        # get train loss
        pass
        # get val loss
        pass
