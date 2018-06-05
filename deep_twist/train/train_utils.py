import torchvision
from deep_twist.data import utils, dataset, transforms
from skimage import io

train_transform = torchvision.transforms.Compose([transforms.CenterCrop(351),
                                                  transforms.RandomRotate(0, 360),
                                                  transforms.CenterCrop(321),
                                                  transforms.RandomTranslate(50),
                                                  transforms.Resize(227)])

train_dataset = dataset.CornellGraspDataset(root_dir='cornell/', 
                                            transform=train_transform)
