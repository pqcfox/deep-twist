import time
import torch
import torchvision
from deep_twist.data import utils, dataset, transforms
from deep_twist.evaluate import utils as eval_utils
from skimage import io
import shutil


def train_model(args, model, loss, train_loader, val_loader, optimizer):
    model.train()
    best_acc = 0.0
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_acc = 0.0
        for batch, (rgd, _, pos) in enumerate(train_loader):
            rgd = rgd.to(args.device)
            pos = [rect.to(args.device) for rect in pos]
            optimizer.zero_grad()
            output = model(rgd)
            loss_val = loss(output, pos)
            loss_val.backward()
            running_loss += loss_val * rgd.size(0)
            rects = utils.one_hot_to_rects(*output)
            # for i in range(rgd.size(0)): # TODO: REEEEMOVE
            #     img = (rgd[i].permute((1, 2, 0))).long()
            #     rect_img = utils.draw_rectangle(img, rects[i], highlight=True)
            #     for j in range(len(pos)):
            #         rect_img = utils.draw_rectangle(rect_img, pos[j][i, :])
            #     io.imsave('whoa-{}-{}.png'.format(i, epoch), rect_img)
            num_correct = eval_utils.count_correct(rects, pos)
            running_acc += num_correct
            optimizer.step()
            if batch % args.log_interval == 0:
                print('[TRAIN] Epoch {}/{}, Batch {}/{}, Loss: {}, Acc: {}'.format(epoch + 1, 
                    args.epochs, batch + 1, len(train_loader), 
                    running_loss / ((batch + 1) * args.batch_size),
                    running_acc / ((batch + 1) * args.batch_size)))
        if (epoch + 1) % args.val_interval == 0:
            accuracy = eval_utils.eval_model(args, model, val_loader)
            print('[VAL] Acc: {}'.format(accuracy))
            torch.save(model.state_dict(), 'checkpoint.pth.tar')
            if accuracy > best_acc:
                best_acc = accuracy
                shutil.copyfile('checkpoint.pth.tar', 'best_model.pth.tar')
