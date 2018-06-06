import torchvision
from deep_twist.data import utils, dataset, transforms
from deep_twist.evaluate import utils as eval_utils
from skimage import io


def train_model(args, model, loss, train_loader, val_loader, optimizer):
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_acc = 0.0
        for batch, (rgd, _, pos) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(rgd)
            loss_val = loss(output, pos)
            loss_val.backward()
            running_loss += loss_val * rgd.size(0)
            rects = utils.one_hot_to_rects(*output)
            num_correct = eval_utils.count_correct(rects, pos)
            running_acc += num_correct * rgd.size(0) 
            optimizer.step()
            if batch % args.log_interval == 0:
                print('[TRAIN] Epoch {}/{}, Batch {}/{}, Loss: {}, Acc: {}'.format(epoch + 1, 
                    args.epochs, batch + 1, len(train_loader), 
                    running_loss / ((batch + 1) * args.batch_size),
                    running_acc / ((batch + 1) * args.batch_size)))
        if (epoch + 1) % args.val_interval == 0:
            accuracy = eval_utils.eval_model(args, model, val_loader)
            # do model saving here...
            print('[VAL] Acc: {}'.format(accuracy))

