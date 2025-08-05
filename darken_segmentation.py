import torch
from torch import nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import loss_func
import numpy as np
from torchvision import transforms
import torch
from utils.data import ImageDataset
from utils.utils import *
from utils.resnet import resnet101
from model import enhance_net_nopool_v2 as enhance_net_nopool # we use v2 in our paper (quadratic curve)

def train(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id #OK
    scale_factor = args.scale_factor #??
    DCE_net = enhance_net_nopool(scale_factor, curve_round=args.curve_round, #OK ?
                                 encode_dim=args.encode_dim,
                                 down_scale=args.down_scale).cuda()

    DCE_net.train() #OK

    train_trans = transforms.Compose([ #Improve
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(args.lowlight_images_path, transform=train_trans) #CHANGE

    train_loader = torch.utils.data.DataLoader( #OK
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    ## REMOVE ##
    # L_color = loss_func.L_color()
    # L_down = loss_func.L_down()
    # L_exp = loss_func.L_exp(args.patch_size, args.exp_weight)
    # L_tv = loss_func.L_TV(mid_val=args.tv_mid)
    ## REMOVE ##

    # if args.sim: #REMOVE
    L_sim = loss_func.L_sim() #CHANGE ?
    resnet = resnet101(pretrained=False, return_features=True).cuda().eval() #CHANGE
    state_dict = torch.load(args.sim_model_dir)['model_state_dict'] #OK ?
    state_dict_resnet = {} #CHANGE
    for k, v in state_dict.items(): #OK
        if 'resnet' in k: #CHANGE
            k_new = k.replace('resnet.', '') #CHANGE
        if 'module' in k_new: #OK ?
            k_new = k_new.replace('module.', '') #OK ?
        state_dict_resnet[k_new] = v #CHANGE
    resnet.load_state_dict(state_dict_resnet) #CHANGE
    resnet.requires_grad_ = False #CHANGE
    # else: #REMOVE
    #     L_sim = None #REMOVE

    optimizer = torch.optim.Adam(DCE_net.parameters( #OK
        ), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR( #IMPROVE
        optimizer, milestones=[int(i) for i in args.decreasing_lr.split(',')], gamma=0.1)

    DCE_net.train() #OK 
    # low_exp, high_exp = args.exp_range.split('-')
    # low_exp, high_exp = float(low_exp), float(high_exp)

    for epoch in range(1, args.num_epochs+1): #OK
        # ltv, ldown, lcol, lexp, lsim = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter() #REMOVE
        lsim_main = AverageMeter()
        lsim_sec = AverageMeter()
        ltotal = AverageMeter()
        for iteration, img in enumerate(train_loader): #OK
            
            # exp = torch.rand(args.train_batch_size, 1, 1, 1).cuda() * (high_exp - low_exp) + low_exp #REMOVE

            img = img.cuda() #OK
            # img = torch.clamp(img, 0, max=args.max_value) #REMOVE

            darkened_img, [enhance_r, down_scale] = DCE_net(img, exp) #CHANGE
            # real_exp = darkened_img[0].mean().item() #REMOVE
            ori_exp = img[0].mean().item() #REMOVE
            if iteration % 50 == 0: #CHANGE FOR WANDB
                torchvision.utils.save_image(
                    darkened_img[0], 'checkpoints/'+args.experiment+'/outputs/'+str(epoch)+'_'+str(iteration)+f'-{(exp[0].item()):.2f}-{real_exp:.2f}.png')
                torchvision.utils.save_image(
                    img[0], 'checkpoints/'+args.experiment+'/outputs/'+str(epoch)+'_'+str(iteration)+f'-gt-{ori_exp:.2f}.png')

            loss = 0. #OK
            ## REMOVE ##
            # loss_TV = args.tv_weight * L_tv(enhance_r)
            # loss += loss_TV
            # ltv.update(loss_TV.item(), args.train_batch_size)

            # loss_col = args.color_weight*torch.mean(L_color(darkened_img))
            # loss += loss_col
            # lcol.update(loss_col.item(), args.train_batch_size)

            # loss_down = args.down_weight*L_down(down_scale)
            # loss += loss_down
            # ldown.update(loss_down.item(), args.train_batch_size)
            ## REMOVE ##

            # if L_sim is not None: #REMOVE
            out_ori = resnet(img) #CHANGE
            out_low = resnet(darkened_img) #CHANGE
            loss_sim = L_sim(list(out_ori), list(out_low)) * args.sim_weight #CHANGE
            loss += loss_sim #CHANGE
            lsim.update(loss_sim.item(), args.train_batch_size) #CHANGE
            # else:#REMOVE
            #     loss_sim = torch.zeros(1)#REMOVE

            ## REMOVE ##
            # loss_exp = torch.mean(L_exp(darkened_img, exp))
            # loss += loss_exp
            # lexp.update(loss_exp.item(), args.train_batch_size)
            ## REMOVE ##
            
            ltotal.update(loss.item(), args.train_batch_size)

            optimizer.zero_grad() #OK
            loss.backward() #OK
            torch.nn.utils.clip_grad_norm( #OK
                DCE_net.parameters(), args.grad_clip_norm)
            optimizer.step() #OK

            if ((iteration+1) % args.display_iter) == 0: #CHANGE TO WANDB
                log.info('Epoch [{}/{}], Iter [{}/{}] Loss: {:.4f} Loss_TV: {:.4f} Loss_down: {:.4f} Loss_col: {:.4f} Loss_exp: {:.4f}, Loss_sim {:4f}'.format(
                            epoch, args.num_epochs, iteration+1, len(train_loader), loss.item(), loss_TV.item(), loss_down.item(), loss_col.item(), loss_exp.item(), loss_sim.item()))

        if  epoch % 5 == 0: #OK
            torch.save(DCE_net.state_dict(
                ), 'checkpoints/' + args.experiment + "/Epoch" + str(epoch) + '.pth')

        scheduler.step() #OK


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str,
                        default='../Cityscapes/leftImg8bit/train',
                        help='path to low-light images'
                        )
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--decreasing_lr', default='5,10', type=str)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--experiment', type=str,
                        required=True, help="Name of the folder where the checkpoints will be saved")
    parser.add_argument('--exp_range', type=str, default='0-0.5')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--exp_weight', type=float, default=10)

    # parser.add_argument('--max_value', type=float, default=1)
    # parser.add_argument('--color_weight',type=float,default=25)

    # parser.add_argument('--tv_weight',type=float,default=1600)
    # parser.add_argument('--tv_mid', type=float, default=0.02)
    # parser.add_argument('--sim',action='store_true')
    parser.add_argument('--sim_weight',type=float,default=0.1)
    parser.add_argument('--sim_model_dir', type=str, default='../segmentation/checkpoints/baseline_RefineNet/best_weights.pth.tar')
    parser.add_argument('--curve_round',type=int,default=8)

    parser.add_argument('--encode_dim',type=int,default=1)
    parser.add_argument('--down_scale',type=float,default=0.95)
    # parser.add_argument('--down_weight',type=float,default=1)

    args = parser.parse_args()

    if not os.path.exists(os.path.join('checkpoints', args.experiment)):
        os.makedirs(os.path.join('checkpoints', args.experiment))
    if not os.path.exists(os.path.join('checkpoints', args.experiment, 'outputs')):
        os.makedirs(os.path.join('checkpoints', args.experiment, 'outputs'))

    log = logger(os.path.join('checkpoints', args.experiment))
    log.info(str(args))

    train(args)
