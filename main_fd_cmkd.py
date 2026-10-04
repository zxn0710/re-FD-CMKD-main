import argparse
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.CremadDataset import CremadDataset
from models.basic_model import AClassifier, VClassifier
from utils.utils import setup_seed, weight_init, log, adjust_lr, norm_feat, FDFilter, Classifier
import time
import tqdm
        
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, default='CREMAD',
                        help='VGGSound, CREMAD, AVE')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    return parser.parse_args()

def train_epoch(args,epoch,a_s_model,v_s_model,a_t_model,v_t_model,filter1,filter2,cls_align_a_low,cls_align_a_high,cls_align_v_low,cls_align_v_high,cls_task_a_low,cls_task_v_low,cls_task_a_high,cls_task_v_high,device,dataloader,optimizer):
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    a_s_model.train()
    v_s_model.train()
    filter1.train()
    filter2.train()
    cls_align_a_low.train()
    cls_align_a_high.train()
    cls_align_v_low.train()
    cls_align_v_high.train()
    cls_task_a_low.train()
    cls_task_v_low.train()
    cls_task_a_high.train()
    cls_task_v_high.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    lr = adjust_lr(iter=epoch, optimizer=optimizer)
    for step, (spec, image, label) in enumerate(tqdm.tqdm(dataloader)):

        spec = spec.to(device)  
        image = image.to(device)  
        label = label.to(device) 
        B = label.shape[0]
        
        with torch.no_grad():
            a_t_model.eval()
            v_t_model.eval()
            a_t,out1 = a_t_model(spec.unsqueeze(1).float())
            v_t,out2 = v_t_model(image.float(), B)

        optimizer.zero_grad()

        a_s,out11 = a_s_model(spec.unsqueeze(1).float())
        v_s,out22 = v_s_model(image.float(), B)

        a_t_low,a_t_high = filter1(a_t)
        v_t_low,v_t_high = filter2(v_t)
        a_s_low,a_s_high = filter1(a_s)
        v_s_low,v_s_high = filter2(v_s)
        
        loss_kd_a_low = criterion2(norm_feat(v_t_low).detach(), norm_feat(a_s_low))
        loss_kd_a_high = criterion2(log(norm_feat(v_t_high)).detach(), log(norm_feat(a_s_high)))
        loss_kd_v_low = criterion2(norm_feat(a_t_low).detach(), norm_feat(v_s_low))
        loss_kd_v_high = criterion2(log(norm_feat(a_t_high)).detach(), log(norm_feat(v_s_high)))
                
        loss_align_a_low = criterion(cls_align_a_low(a_s_low), label) + criterion(cls_align_a_low(v_t_low), label)
        loss_align_a_high = criterion(cls_align_a_high(a_s_high), label) + criterion(cls_align_a_high(v_t_high), label)
        loss_align_v_low = criterion(cls_align_v_low(v_s_low), label) + criterion(cls_align_v_low(a_t_low), label)
        loss_align_v_high = criterion(cls_align_v_high(v_s_high), label) + criterion(cls_align_v_high(a_t_high), label)
        
        loss_kd_a = loss_kd_a_low + loss_kd_a_high 
        loss_kd_v = loss_kd_v_low + loss_kd_v_high 

        loss_align_a = loss_align_a_low + loss_align_a_high
        loss_align_v = loss_align_v_low + loss_align_v_high
        
        loss_task_a = criterion(out11, label) + criterion(cls_task_a_low(a_s_low), label) + criterion(cls_task_a_high(a_s_high), label)
        loss_task_v = criterion(out22, label) + criterion(cls_task_v_low(v_s_low), label) + criterion(cls_task_v_high(v_s_high), label)
      
        loss_a = loss_kd_a + loss_align_a + loss_task_a
        loss_v = loss_kd_v + loss_align_v + loss_task_v
        loss = loss_a + loss_v

        loss.backward()
        optimizer.step()
        
        _loss += loss.item()
        _loss_a += loss_a.item() 
        _loss_v += loss_v.item()  
    
    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args,a_s_model,v_s_model,filter1,filter2,cls_task_a_low,cls_task_v_low,cls_task_a_high,cls_task_v_high,device,dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 50
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        a_s_model.eval()
        v_s_model.eval()
        filter1.eval()
        filter2.eval()
        cls_task_a_low.eval()
        cls_task_v_low.eval()
        cls_task_a_high.eval()
        cls_task_v_high.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc1 = [0.0 for _ in range(n_classes)]
        acc2 = [0.0 for _ in range(n_classes)]
        acca1 = [0.0 for _ in range(n_classes)]
        acca2 = [0.0 for _ in range(n_classes)]
        accv1 = [0.0 for _ in range(n_classes)]
        accv2 = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            B = label.shape[0]
            
            a_s,out1 = a_s_model(spec.unsqueeze(1).float())
            v_s,out2 = v_s_model(image.float(), B)

            a_s_low,a_s_high = filter1(a_s)
            v_s_low,v_s_high = filter2(v_s)

            out_a1 = cls_task_a_low(a_s_low)
            out_a2 = cls_task_a_high(a_s_high)
            out_v1 = cls_task_v_low(v_s_low)
            out_v2 = cls_task_v_high(v_s_high)
        
            prediction1 = softmax(out1)
            prediction2 = softmax(out2)
            predictiona1 = softmax(out_a1)
            predictiona2 = softmax(out_a2)
            predictionv1 = softmax(out_v1)
            predictionv2 = softmax(out_v2)
            for i in range(image.shape[0]):
                ma1 = np.argmax(prediction1[i].cpu().data.numpy())
                ma2 = np.argmax(prediction2[i].cpu().data.numpy())
                maa1 = np.argmax(predictiona1[i].cpu().data.numpy())
                maa2 = np.argmax(predictiona2[i].cpu().data.numpy())
                mav1 = np.argmax(predictionv1[i].cpu().data.numpy())
                mav2 = np.argmax(predictionv2[i].cpu().data.numpy())
                num[label[i]] += 1.0  
                if np.asarray(label[i].cpu()) == ma1:
                    acc1[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == ma2:
                    acc2[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == maa1:
                    acca1[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == maa2:
                    acca2[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == mav1:
                    accv1[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == mav2:
                    accv2[label[i]] += 1.0
        print('acc_a1:',sum(acca1) / sum(num),'acc_a2:',sum(acca2) / sum(num),'acc_v1:',sum(accv1) / sum(num),'acc_v2:',sum(accv2) / sum(num))
    return sum(acc1) / sum(num), sum(acc2) / sum(num), sum(acca1) / sum(num), sum(acca2) / sum(num),sum(accv1) / sum(num), sum(accv2) / sum(num)


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    a_t_model = AClassifier(args)
    v_t_model = VClassifier(args)
        
    a_t_model.load_state_dict(torch.load('./ckpt/unimodality-audio/model-CREMAD-frm1-bsz64-lr0.01/best.pt')['model'])
    v_t_model.load_state_dict(torch.load('./ckpt/unimodality-visual/model-CREMAD-frm1-bsz64-lr0.01/best.pt')['model'])
            
    a_t_model.to(device)
    v_t_model.to(device)  

    a_s_model = AClassifier(args)
    v_s_model = VClassifier(args)
    
    a_s_model.apply(weight_init)
    a_s_model.to(device)
    v_s_model.apply(weight_init)
    v_s_model.to(device)

    filter1 = FDFilter()
    filter2 = FDFilter()

    filter1.to(device)
    filter2.to(device)

    cls_align_a_low = Classifier(512, 6)
    cls_align_a_high = Classifier(512, 6)
    cls_align_v_low = Classifier(512, 6)
    cls_align_v_high = Classifier(512, 6)
    cls_task_a_low = Classifier(512, 6)
    cls_task_v_low = Classifier(512, 6)
    cls_task_a_high = Classifier(512, 6)
    cls_task_v_high = Classifier(512, 6)
    cls_align_a_low.to(device)
    cls_align_a_high.to(device)
    cls_align_v_low.to(device)
    cls_align_v_high.to(device)
    cls_task_a_low.to(device)
    cls_task_v_low.to(device)
    cls_task_a_high.to(device)
    cls_task_v_high.to(device)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(list(a_s_model.parameters()) + list(v_s_model.parameters())+list(filter1.parameters()) + list(filter2.parameters())+list(cls_align_a_low.parameters()) + list(cls_align_a_high.parameters())+list(cls_align_v_low.parameters()) + list(cls_align_v_high.parameters())+list(cls_task_a_low.parameters()) + list(cls_task_v_low.parameters())+list(cls_task_a_high.parameters()) + list(cls_task_v_high.parameters()), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(list(a_s_model.parameters()) + list(v_s_model.parameters())+list(filter1.parameters()) + list(filter2.parameters())+list(cls_align_a_low.parameters()) + list(cls_align_a_high.parameters())+list(cls_align_v_low.parameters()) + list(cls_align_v_high.parameters())+list(cls_task_a_low.parameters()) + list(cls_task_v_low.parameters())+list(cls_task_a_high.parameters()) + list(cls_task_v_high.parameters()), lr=args.learning_rate, betas=(0.9, 0.99))

    train_dataset = CremadDataset(args, mode='train')
    test_dataset = CremadDataset(args, mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=24,
                                  shuffle=True, pin_memory=True) 

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=24,
                                 shuffle=False, pin_memory=False)

    if args.train:
        trainloss_file = args.logs_path + '/FD_CMKD' + '/train_loss-' + args.dataset + '-bsz' + \
                         str(args.batch_size) + '-lr' + str(args.learning_rate) \
                         + '-epoch' + str(args.epochs) + '.txt'
        if not os.path.exists(args.logs_path + '/FD_CMKD'):
            os.makedirs(args.logs_path + '/FD_CMKD')

        save_path = args.ckpt_path + '/FD_CMKD' + '/model-' + args.dataset + '-bsz' + \
                    str(args.batch_size) + '-lr' + str(args.learning_rate) \
                    + '-epoch' + str(args.epochs)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file) 
        f_trainloss = open(trainloss_file, 'a')

        best_acc1 = 0.0
        best_acc2 = 0.0

        for epoch in range(args.epochs):
            print('Epoch: {}: '.format(epoch))

            s_time = time.time()
            batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, a_s_model,v_s_model,
                                                                 a_t_model,v_t_model,filter1,filter2,cls_align_a_low,cls_align_a_high,cls_align_v_low,cls_align_v_high,cls_task_a_low,cls_task_v_low,cls_task_a_high,cls_task_v_high,
                                                                 device,
                                                                 train_dataloader,
                                                                 optimizer)
            e_time = time.time()
            print('per epoch time: ', e_time - s_time)
            acc_a, acc_v, acc_a1, acc_a2, acc_v1, acc_v2 = valid(args, a_s_model, v_s_model,filter1,filter2,cls_task_a_low,cls_task_v_low,cls_task_a_high,cls_task_v_high, device, test_dataloader)
            print('epoch: ', epoch, 'loss: ', batch_loss, batch_loss_a, batch_loss_v)
            print('epoch: ', epoch, 'acc_a: ', acc_a, 'acc_v: ', acc_v)
            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(batch_loss_a) +
                              "\t" + str(batch_loss_v) +
                              "\t" + str(acc_a) +
                              "\t" + str(acc_v) +
                              "\t" + str(acc_a1) +
                              "\t" + str(acc_a2) +
                              "\t" + str(acc_v1) +
                              "\t" + str(acc_v2) +
                              "\n")

            if max(acc_a,acc_a1,acc_a2) > best_acc1:
                if max(acc_a,acc_a1,acc_a2) > best_acc1:
                    best_acc1 = float(max(acc_a,acc_a1,acc_a2))
                print('Saving model....')

                torch.save(
                    {
                        'model': a_s_model.state_dict(),
                        'cls_task_a_low': cls_task_a_low.state_dict(),
                        'cls_task_a_high': cls_task_a_high.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(save_path, 'best_a.pt'.format(epoch))
                )
                print('Saved a_s_model!!!')
                
            if max(acc_v,acc_v1,acc_v2) > best_acc2:
                if max(acc_v,acc_v1,acc_v2) > best_acc2:
                    best_acc2 = float(max(acc_v,acc_v1,acc_v2))
                print('Saving model....')

                torch.save(
                    {
                        'model': v_s_model.state_dict(),
                        'cls_task_a_low': cls_task_a_low.state_dict(),
                        'cls_task_a_high': cls_task_a_high.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(save_path, 'best_v.pt'.format(epoch))
                )
                print('Saved v_s_model!!!')
                
        print('best_a:',best_acc1,'best_v:',best_acc2)
        f_trainloss.write(str(best_acc1) +
                              "\t" + str(best_acc2) +
                              "\n")
        f_trainloss.flush()


if __name__ == "__main__":
    main()
