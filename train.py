import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.functional as F
from torchvision.transforms import functional as F2
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from model import BuildAlexNet, BuildCBAMAlexNetAll
import cv2
from torchvision.transforms import transforms
from PIL import Image
import random


def color_transform_generator(brightness_factor, contrast_factor, saturation_factor, hue_factor):
    """Get a randomized transform to be applied on image.

    Arguments are same as that of __init__.

    Returns:
        Transform which randomly adjusts brightness, contrast and
        saturation in a random order.
    """
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: F2.adjust_brightness(img, brightness_factor)))
    transform_list.append(transforms.Lambda(lambda img: F2.adjust_contrast(img, contrast_factor)))
    transform_list.append(transforms.Lambda(lambda img: F2.adjust_saturation(img, saturation_factor)))
    transform_list.append(transforms.Lambda(lambda img: F2.adjust_hue(img, hue_factor)))

    random.shuffle(transform_list)
    transform = transforms.Compose(transform_list)

    return transform


class MpImageDataset(Dataset):

    def __init__(self, label_path, img_path, data_path=None, is_training=False, distance_list=['near', 'mid', 'far', 'all']):
        self.labels = pd.read_csv(label_path)
        if not is_training:
            labels_1 = self.labels[self.labels['label'] == 1]
            labels_0 = self.labels[self.labels['label'] == 0].sample(len(labels_1), random_state=8)
            self.labels = labels_0.append(labels_1, True)
            print(len(labels_0), len(labels_1), len(self.labels))
        self.labels = {
            self.labels['num'][i]: self.labels['label'][i] for i in range(len(self.labels))
        }
        self.img_path = img_path
        self.data_path = data_path
        self.spd_df = pd.read_csv("./spd_total.csv")
        self.file_list = list(self.labels.keys())
        # print(self.labels)
        self.len = len(self.file_list)
        self.is_training = is_training
        self.distance_list = distance_list

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        
        self.random_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
            # transforms.RandomGrayscale(p=0.25)
        ])
        

    def __getitem__(self, index):
        # if os.path.exists(f'{self.img_path}/{self.file_list[index]}.jpg'):
        #     x = Image.open(f'{self.img_path}/{self.file_list[index]}.jpg')
        # else:
        #     x = Image.open(f'/home/xujingning/data/VOLVO_model/img/valid_mp_img2/{self.file_list[index]}.jpg')
        
        mps = []
        if False and self.is_training:
            is_flip = False
            is_color_flip = False
            if random.random() < 0.5:
                is_flip = True
            
            if random.random() < 0.5:
                is_color_flip = True
            
            random_factor = 0.6
            brightness_factor = random.uniform(1 - random_factor, 1 + random_factor)

            contrast_factor = random.uniform(1 - random_factor, 1 + random_factor)

            saturation_factor = random.uniform(1 - random_factor, 1 + random_factor)

            hue_factor = random.uniform(-0.5, 0.5)

            color_transform = color_transform_generator(brightness_factor, contrast_factor, saturation_factor, hue_factor)
            
        for distance in self.distance_list:
            if distance == 'all':
                x = Image.open(f'/data/yolo_motion/yolo/yolo_motion/mp/{self.file_list[index]}.avi.jpg')
            else:
                x = Image.open(f'/home/xujingning/data/VOLVO/img/{distance}/{self.file_list[index]}.png')
            # x = x.crop((0,0,656, 30))
            x = x.crop((0,0,656, 48))
            if False and self.is_training:
                # x = self.random_transform(x)
                if is_flip:
                    x = transforms.RandomHorizontalFlip(1)(x)
                x = color_transform(x)
            # if index <= 10 and self.is_training:
            #     x.save(f"saveimg/{self.file_list[index]}_{distance}.png")
            x = self.img_transform(x)
            x = 1 - x
            if self.is_training:
                # if is_color_flip:
                #     x = 1 - x
                noise = torch.randn(x.size()) * 0.05
                # print(noise)
                x += noise
                x = torch.clamp(x, 0, 1).detach()

            mps.append(x)

        x = torch.cat(mps, dim=0)
        # print(x.size())


        
        # x = cv2.resize(x, )
        # print(self.file_list[index].split('.')[0], sum(sum(np.isnan(x))))
        # print(self.file_list[index].split('.')[0], self.labels)
        y = self.labels[int(self.file_list[index])]
        # print(self.file_list[index], x.shape)
        if self.data_path is not None:
            data = np.load(f"{self.data_path}/{self.file_list[index]}.npy").astype(np.float32)
            # data = self.spd_df.loc[self.spd_df['num'] == self.file_list[index], [f'speed_{kk}s' for kk in range(13)]].iloc[0]

            # data = self.spd_df.loc[self.spd_df['num'] == self.file_list[index], 'speed_7s'].iloc[0]
            # data = np.array(data).astype(np.float32)

            # data = data - data.mean()
            # _kurt, _skew = data.kurt(), data.skew()
            # data = np.array(data).astype(np.float32)
            # _mean, _std = np.mean(data), np.std(data)
            # _max, _min = data.max(), data.min()
            # data = np.append(data, _mean)
            # data = np.append(data, _std)
            # data = np.append(data, _kurt)
            # data = np.append(data, _skew)
            # data = np.append(data, _max)
            # data = np.append(data, _min)
            # data = (data - np.mean(data)) / np.std(data)
            # data = (data - np.mean(data))
            # print(data, np.mean(data), np.std(data))
            # print(data.reshape(-1))

            x = [x, data]
            # x = [x, data.reshape(-1)]
        if self.is_training:
            return x, y
        else:
            return x, y, self.file_list[index]

    def __len__(self):
        return self.len

def roc_auc_score(true, scores):
    count = 0
    data = list(zip(true, scores))
    positives = list(filter(lambda x: x[0] == 1, data))
    negatives = list(filter(lambda x: x[0] == 0, data))

    for positive in positives:
        for negative in negatives:
            if positive[1] > negative[1]:
                count += 1
            elif positive[1] == negative[1]:
                count += 0.5
    return count / (len(positives) * len(negatives))

def get_acc(output, label):
    total = output.shape[0]
    pred_label = torch.ones(output.size()).long().cuda()
    # print(pred_label)
    # print(output.mean())
    pred_label[output < 0.5] = 0
    pred_label = pred_label.view(-1)
    # print(pred_label)
    num_correct = (pred_label == label).sum().item()
    TP = ((pred_label == label)&(label == 1)).sum().item()
    TN = ((pred_label == label)&(label == 0)).sum().item()
    FP = ((pred_label != label)&(pred_label == 1)).sum().item()
    FN = ((pred_label != label)&(pred_label == 0)).sum().item()
    return num_correct / total, TP, TN, FP, FN

def mean(arr):
    return sum(arr) / len(arr)

def main():
    TRAIN_LABEL_PATH = '/home/xujingning/data/VOLVO_model/label/train_label_new3.csv'
    TEST_LABEL_PATH = '/home/xujingning/data/VOLVO_model/label/test_label_new.csv'
    VALIDATE_LABEL_PATH = '/home/xujingning/data/VOLVO_model/label/valid_label_new.csv'
    TRAIN_DATA_PATH = '/home/xujingning/data/VOLVO_model/data/data_new2/total/'
    TEST_DATA_PATH = '/home/xujingning/data/VOLVO_model/data/data_new2/total/'
    VALIDATE_DATA_PATH = '/home/xujingning/data/VOLVO_model/data/data_new2/total/'
    # TRAIN_PATH = '/home/xujingning/data/VOLVO_model/img/train_img/'
    # TEST_PATH = '/home/xujingning/data/VOLVO_model/img/test_img/'
    # VALIDATE_PATH = '/home/xujingning/data/VOLVO_model/img/validate_img/'
    TRAIN_PATH = '/home/xujingning/data/VOLVO_model/img/train_mp_img2/'
    TEST_PATH = '/home/xujingning/data/VOLVO_model/img/test_mp_img2/'
    VALIDATE_PATH = '/home/xujingning/data/VOLVO_model/img/valid_mp_img2/'
    # MODEL_PATH = '/home/xujingning/data/VOLVO_model/torch_resnet_newdata_distance3_test1/'
    # MODEL_PATH = '/home/xujingning/data/VOLVO_model/torch_alexnet_newdata_distance3_pred_test9/'
    # MODEL_PATH = '/home/xujingning/data/VOLVO_model/torch_alexnet_newdata_distance3_pred7_test9/'
    # MODEL_PATH = '/home/xujingning/data/VOLVO_model/torch_alexnet_newdata_distance3_at_all_test8/'
    MODEL_PATH = '/home/xujingning/data/VOLVO_model/torch_alexnet_distance3_wodata_test3/'


    distance_list =[ 'near', 'mid', 'far']
    use_kinematic = True
    dim_kinematic = 1 if use_kinematic else 0

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    net = BuildCBAMAlexNetAll(model_type='new', n_input=len(distance_list) * 3, n_output=1, dim_kinematic=dim_kinematic)

    train_dataset = MpImageDataset(TRAIN_LABEL_PATH, TRAIN_PATH, is_training=True, distance_list=distance_list, data_path=TRAIN_DATA_PATH)
    test_dataset = MpImageDataset(TEST_LABEL_PATH, TEST_PATH, distance_list=distance_list, data_path=TRAIN_DATA_PATH)
    valid_dataset = MpImageDataset(TEST_LABEL_PATH, TEST_PATH, distance_list=distance_list, data_path=TRAIN_DATA_PATH)
    


    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=8,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=512,
                              num_workers=8,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=512,
                              num_workers=8,
                              shuffle=True)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)
    # params = list(net.classifier.parameters())
    # params += list(net.lstm.parameters())
    # optimizer = torch.optim.SGD(params, lr=1e-2, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    test_best_auc = 0
    valid_best_auc = 0

    net.cuda()
    for epoch in range(3000):
        train_loss = []
        train_acc = []
        
        # print(net.classifier[0].weight.data, net.classifier[0].bias.data)
        if epoch % 1 == 0:
            tTP, tTN, tFP, tFN = 0, 0, 0, 0
            valid_loss, valid_acc = [], []
            scores = torch.FloatTensor().cuda()
            real_labels = torch.LongTensor().cuda()
            net.eval()
            event_list = []
            for i, data in enumerate(valid_loader):
                inputs, labels, events = data
                # print(events)
                event_list += list(events)
                # print(inputs.size())
                if use_kinematic:
                    inputs, kinematic = inputs
                    kinematic = torch.FloatTensor(kinematic)
                    inputs, kinematic, labels = Variable(inputs), Variable(kinematic), Variable(labels)
                    inputs, kinematic, labels = inputs.cuda(), kinematic.cuda(), labels.cuda()

                    with torch.no_grad():
                        output = net(inputs, kinematic, event_list)

                # 将这些数据转换成Variable类型
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    inputs, labels = inputs.cuda(), labels.cuda()
                    with torch.no_grad():
                        output = net(inputs)

                clf_res = torch.sigmoid(output)
                # clf_res = torch.softmax(output, dim=1)[:,1]
                scores = torch.cat([scores, clf_res.view(-1)], dim=0)
                real_labels = torch.cat([real_labels, labels.view(-1)], dim=0)

                loss = criterion(output.view(-1), labels.float())

                acc, TP, TN, FP, FN = get_acc(clf_res, labels)
                tTP += TP
                tTN += TN
                tFP += FP
                tFN += FN
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            
            auc = roc_auc_score(real_labels, scores)
            print(f'Validating... Epoch: {epoch} loss: {mean(valid_loss)} '
                    f'TP: {tTP} TN: {tTN} FP: {tFP} FN: {tFN} acc: {(tTP + tTN)/ (tTP + tTN + tFP + tFN)}  precision: '
                    f'{tTP / (tTP + tFP) if tTP + tFP > 0 else "nan"} recall: {tTP / (tTP + tFN)}'
                    f' score 0: {scores[real_labels == 0].mean().item()} score 1: {scores[real_labels == 1].mean().item()} AUC: {auc}')

            if True or auc > valid_best_auc:
                valid_best_auc = auc
                acc = (tTP + tTN)/ (tTP + tTN + tFP + tFN)
                pd.DataFrame({
                    'event': event_list,
                    'label': real_labels,
                    'score': scores
                }).to_csv(f'{MODEL_PATH}best_valid_model-epoch{epoch}-TP{tTP}-TN-{tTN}-FP-{tFP}-FN-{tFN}-auc{valid_best_auc:.5f}-acc{acc:.5f}.csv', index=False)
                torch.save(net.state_dict(), f'{MODEL_PATH}best_valid_model-TP{tTP}-TN-{tTN}-FP-{tFP}-FN-{tFN}-auc{valid_best_auc:.5f}-acc{acc:.5f}.pth')



        net.train()
        tTP, tTN, tFP, tFN = 0, 0, 0, 0
        scores = torch.FloatTensor().cuda()
        real_labels = torch.LongTensor().cuda()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # print(inputs.size())
            if use_kinematic:
                inputs, kinematic = inputs
                inputs, kinematic, labels = Variable(inputs), Variable(kinematic), Variable(labels)
                inputs, kinematic, labels = inputs.cuda(), kinematic.cuda(), labels.cuda()

                output = net(inputs, kinematic)

            # 将这些数据转换成Variable类型
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                output = net(inputs)

            clf_res = torch.sigmoid(output)
            # clf_res = torch.softmax(output, dim=1)[:,1]
            scores = torch.cat([scores, clf_res.view(-1)], dim=0)
            real_labels = torch.cat([real_labels, labels.view(-1)], dim=0)

            loss = criterion(output.view(-1), labels.float())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc, TP, TN, FP, FN = get_acc(clf_res, labels)
            tTP += TP
            tTN += TN
            tFP += FP
            tFN += FN
            train_loss.append(loss.item())
            train_acc.append(acc)
        # print(scores, scores.size())
        # print(real_labels, real_labels.size())

        auc = roc_auc_score(real_labels, scores)
        print(f'Training... Epoch: {epoch} loss: {mean(train_loss)} '
              f'TP: {tTP} TN: {tTN} FP: {tFP} FN: {tFN} acc: {(tTP + tTN)/ (tTP + tTN + tFP + tFN)} precision: '
              f'{tTP / (tTP + tFP) if tTP + tFP > 0 else "nan"} recall: {tTP / (tTP + tFN)}'
              f' score 0: {scores[real_labels == 0].mean().item()} score 1: {scores[real_labels == 1].mean().item()} AUC: {auc}')


        if epoch % 5 == 6:
            tTP, tTN, tFP, tFN = 0, 0, 0, 0
            test_loss, test_acc = [], []
            scores = torch.FloatTensor().cuda()
            real_labels = torch.LongTensor().cuda()
            net.eval()
            for i, data in enumerate(test_loader):
                inputs, labels = data
                if use_kinematic:
                    inputs, kinematic = inputs
                    inputs, kinematic, labels = Variable(inputs), Variable(kinematic), Variable(labels)
                    inputs, kinematic, labels = inputs.cuda(), kinematic.cuda(), labels.cuda()

                    with torch.no_grad():
                        output = net(inputs, kinematic)

                # 将这些数据转换成Variable类型
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    inputs, labels = inputs.cuda(), labels.cuda()

                    with torch.no_grad():
                        output = net(inputs)

                clf_res = torch.sigmoid(output)
                # clf_res = torch.softmax(output, dim=1)[:,1]
                scores = torch.cat([scores, clf_res.view(-1)], dim=0)
                real_labels = torch.cat([real_labels, labels.view(-1)], dim=0)

                loss = criterion(output.view(-1), labels.float())

                acc, TP, TN, FP, FN = get_acc(clf_res, labels)
                tTP += TP
                tTN += TN
                tFP += FP
                tFN += FN
                test_loss.append(loss.item())
                test_acc.append(acc)
            auc = roc_auc_score(real_labels, scores)
            print(f'Testting...  loss: {mean(test_loss)} '
                  f'TP: {tTP} TN: {tTN} FP: {tFP} FN: {tFN} acc: {(tTP + tTN) / (tTP + tTN + tFP + tFN)} precision: '
                  f'{tTP / (tTP + tFP) if tTP + tFP > 0 else "nan"} recall: {tTP / (tTP + tFN)}'
                  f' score 0: {scores[real_labels == 0].mean().item()} score 1: {scores[real_labels == 1].mean().item()} AUC: {auc}')
            if auc > test_best_auc:
                test_best_auc = auc
                # torch.save(net.state_dict(), f'{MODEL_PATH}best_test_model-{test_best_auc:.5f}.pth')
                acc = (tTP + tTN)/ (tTP + tTN + tFP + tFN)
                torch.save(net.state_dict(), f'{MODEL_PATH}best_test_model-auc{test_best_auc:.5f}-acc{acc:.5f}.pth')
                    

    tTP, tTN, tFP, tFN = 0, 0, 0, 0
    test_loss, test_acc = [], []
    scores = torch.FloatTensor().cuda()
    real_labels = torch.LongTensor().cuda()
    net.eval()
    for i, data in enumerate(test_loader):
        inputs, labels = data

        # 将这些数据转换成Variable类型
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        output = net(inputs)

        clf_res = torch.sigmoid(output)
        # clf_res = torch.softmax(output, dim=1)[:,1]
        scores = torch.cat([scores, clf_res.view(-1)], dim=0)
        real_labels = torch.cat([real_labels, labels.view(-1)], dim=0)

        loss = criterion(output.view(-1), labels.float())

        acc, TP, TN, FP, FN = get_acc(clf_res, labels)
        tTP += TP
        tTN += TN
        tFP += FP
        tFN += FN
        test_loss.append(loss.item())
        test_acc.append(acc)
    auc = roc_auc_score(real_labels, scores)
    print(f'Testting...  loss: {mean(test_loss)} '
          f'TP: {tTP} TN: {tTN} FP: {tFP} FN: {tFN} acc: {(tTP + tTN)/ (tTP + tTN + tFP + tFN)} precision: '
          f'{tTP / (tTP + tFP) if tTP + tFP > 0 else "nan"} recall: {tTP / (tTP + tFN)}'
          f' score 0: {scores[real_labels == 0].mean().item()} score 1: {scores[real_labels == 1].mean().item()} AUC: {auc}')

    torch.save(net.state_dict(), f'{MODEL_PATH}model.pth')


if __name__ == '__main__':
    main()
