from torchvision import models
import torch.nn as nn
import torch
# from torchsummary import summary
from cbam import *

# 原AlexNet
class BuildAlexNet(nn.Module):
    def __init__(self, model_type, n_input, n_output, dim_kinematic=0):
        super(BuildAlexNet, self).__init__()
        self.model_type = model_type
        if model_type == 'pre':
            model = models.alexnet(pretrained=True)
            self.features = model.features

            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            fc1 = nn.Linear(9216, 4096)
            fc1.bias = model.classifier[1].bias
            fc1.weight = model.classifier[1].weight

            fc2 = nn.Linear(4096, 4096)
            fc2.bias = model.classifier[4].bias
            fc2.weight = model.classifier[4].weight

            self.classifier = nn.Sequential(
                nn.Dropout(),
                fc1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                fc2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, n_output))
            # 或者直接修改为
        #            model.classifier[6]==nn.Linear(4096,n_output)
        #            self.classifier = model.classifier
        if model_type == 'new':
            '''
            ZNB!!!!!
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 11, 4, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                nn.Conv2d(16, 48, 5, 1, 2),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                nn.Conv2d(48, 96, 3, 1, 1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                # nn.Conv2d(192, 128, 3, 1, 1),
                # nn.BatchNorm2d(128),
                # nn.ReLU(inplace=True),
                nn.Conv2d(96, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0))
            self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
            self.FC6 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(288, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True))
            self.FC7 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(128 + dim_kinematic, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True))
            self.FC8 = nn.Linear(32, n_output)

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(288, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(128 + dim_kinematic, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, n_output))
            '''

            '''
            self.features = nn.Sequential(
                nn.Conv2d(9, 64, 11, 4, 2),
                # nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                nn.Conv2d(64, 192, 5, 1, 2),
                # nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                nn.Conv2d(192, 384, 3, 1, 1),
                # nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0))
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(9216, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, n_output))

            '''


            '''
            self.features = nn.Sequential(
                nn.Conv2d(n_input, 16, 11, 4, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),

                nn.Conv2d(16, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),

                nn.Conv2d(32, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                # nn.Conv2d(64, 64, 3, 1, 1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),

                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0))
            self.avgpool = nn.AdaptiveAvgPool2d((3, 3))


            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(144, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                # nn.Linear(128, 32),
                # nn.BatchNorm1d(32),
                # nn.ReLU(inplace=True),
                nn.Linear(32, n_output))
            '''

            
            '''
            self.features = nn.Sequential(
                nn.Conv2d(n_input, 16, 11, 4, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                # nn.Conv2d(32, 64, 3, 1, 1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(64, 64, 3, 1, 1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0))
            self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(288, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, n_output))

            
            # self.FC6 = nn.Sequential(
            #     nn.Dropout(),
            #     nn.Linear(288, 128),
            #     nn.BatchNorm1d(128),
            #     nn.ReLU(inplace=True))
            # self.FC7 = nn.Sequential(
            #     nn.Dropout(),
            #     nn.Linear(128 + dim_kinematic, 32),
            #     nn.BatchNorm1d(32),
            #     nn.ReLU(inplace=True))
            # self.FC8 = nn.Linear(32, n_output)

            '''

            '''
            self.features = nn.Sequential(
                nn.Conv2d(n_input, 24, 11, 4, 2),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),

                nn.Conv2d(24, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),

                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                # nn.Conv2d(64, 64, 3, 1, 1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),

                nn.Conv2d(64, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0))
            self.avgpool = nn.AdaptiveAvgPool2d((3, 3))


            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(144, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                # nn.Dropout(),
                nn.Linear(32, n_output))

            '''

            self.features = nn.Sequential(
                nn.Conv2d(n_input, 16, 11, 4, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                # nn.Conv2d(32, 64, 3, 1, 1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(64, 64, 3, 1, 1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0))
            self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(288, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, n_output))

    def forward(self, x, data=None, event=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if data is None:
            out = self.classifier(x)
        else:
            x = self.FC6(x)
            # print(x.size())
            # print(data.size())
            x = torch.cat([x, data], dim=1)
            # print(x.size())
            x = self.FC7(x)
            out = self.FC8(x)
        return out

# 轻量级CNN
class BuildAlexNetSimple(nn.Module):
    def __init__(self, model_type, n_input, n_output, dim_kinematic=0):
        super(BuildAlexNetSimple, self).__init__()
        self.model_type = model_type
        if model_type == 'pre':
            model = models.alexnet(pretrained=True)
            self.features = model.features

            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            fc1 = nn.Linear(9216, 4096)
            fc1.bias = model.classifier[1].bias
            fc1.weight = model.classifier[1].weight

            fc2 = nn.Linear(4096, 4096)
            fc2.bias = model.classifier[4].bias
            fc2.weight = model.classifier[4].weight

            self.classifier = nn.Sequential(
                nn.Dropout(),
                fc1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                fc2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, n_output))
        if model_type == 'new':
            self.features = nn.Sequential(
                nn.Conv2d(n_input, 16, 5),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 0))
            self.avgpool = nn.AdaptiveAvgPool2d((1))

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Linear(16, n_output))

    def forward(self, x, data=None, event=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

# 轻量级CNN + CBAM
class BuildCBAMAlexNet(nn.Module):
    def __init__(self, model_type, n_input, n_output, dim_kinematic=0):
        super(BuildCBAMAlexNet, self).__init__()
        model = BuildAlexNet(model_type, n_input, n_output, dim_kinematic)

        a = torch.load("/home/xujingning/data/VOLVO_model/torch_test_newdata_distance3_test5/best_valid_model-TP103-TN-180-FP-35-FN-28-auc0.89870-acc0.81792.pth")
        model.load_state_dict(a)
        # self.features = model.features
        self.avgpool = model.avgpool
        # self.classifier = model.classifier

        self.features_1 = torch.nn.Sequential(*list(model.features.children())[:3])
        self.features_2 = torch.nn.Sequential(*list(model.features.children())[4:7])
        self.features_3 = torch.nn.Sequential(*list(model.features.children())[8:11])
        self.features_4 = torch.nn.Sequential(*list(model.features.children())[11:14])
        self.features = torch.nn.Sequential(
            self.features_1,
            CBAM(24, 16),
            list(model.features.children())[3],
            self.features_2,
            CBAM(32, 16),
            list(model.features.children())[7],
            self.features_3,
            # CBAM(64, 16),
            self.features_4,
            CBAM(16, 16),
            list(model.features.children())[14],
        )
        self.classifier = torch.nn.Sequential(
            *list(model.classifier.children())[:8],
            nn.Dropout(),
            list(model.classifier.children())[8]
        )

    def forward(self, x, data=None, event=None):
        x = self.features(x)
        # x = self.cbam(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out

    
class BuildCNNwithCBAMAndLSTM(nn.Module):
    def __init__(self, model_type, n_input, n_output, dim_kinematic=0):
        super(BuildCBAMandSpeedCNN, self).__init__()
        self.CNN = BuildCBAMAlexNet(model_type, n_input, n_output, dim_kinematic)

        self.classifier = nn.Sequential(
            nn.Linear(3, n_output))
        
        in_feature = 3
        hidden_feature = 16
        num_layers = 3
        num_class = 2
        self.lstm = lstm(in_feature, hidden_feature, num_class, num_layers)

        self.classifier.apply(self.weights_init)
        # self.spd_lr.apply(self.weights_init2)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            # m.weight.data是卷积核参数, m.bias.data是偏置项参数
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            # m.weight.data.normal_(0.0, 0.02)
            m.weight.data = torch.FloatTensor([[1.0, 0.5, -0.5]])
            m.bias.data.zero_()
            print(m.weight.data, m.bias.data)
            
    def weights_init2(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            # m.weight.data是卷积核参数, m.bias.data是偏置项参数
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.zero_()
            print(m.weight.data, m.bias.data)

    def forward(self, x, data=None, event=None):
        x = self.CNN(x)
        data = torch.relu(self.lstm(data))
        x = torch.cat([x, data], dim=1)
        out = self.classifier(x)
        return out

    
# 轻量级CNN + CBAM + 运动学数据(全连接)
class BuildCNNwithCBAMAndFC(nn.Module):
    def __init__(self, model_type, n_input, n_output, dim_kinematic=0):
        super(BuildCBAMAlexNetAll, self).__init__()
        self.model_type = model_type

        if model_type == 'new':
            self.features = nn.Sequential(
                nn.Conv2d(n_input, 16, 11, 4, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                CBAM(16, 8),
                nn.MaxPool2d(3, 2, 0),

                nn.Conv2d(16, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                CBAM(32, 8),
                nn.MaxPool2d(3, 2, 0),

                nn.Conv2d(32, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                CBAM(32, 8),

                # nn.Conv2d(64, 64, 3, 1, 1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
                # CBAM(64, 16),

                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                CBAM(16, 8),
                nn.MaxPool2d(3, 2, 0))
                
            self.avgpool = nn.AdaptiveAvgPool2d((3, 3))


            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(144, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                # nn.Linear(128, 32),
                # nn.BatchNorm1d(32),
                # nn.ReLU(inplace=True),
                nn.Linear(32, n_output))

    def forward(self, x, data=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if data is None:
            out = self.classifier(x)
        else:
            x = self.FC6(x)
            # print(x.size())
            # print(data.size())
            x = torch.cat([x, data], dim=1)
            # print(x.size())
            x = self.FC7(x)
            out = self.FC8(x)
        return out


class lstm(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=2):
        super(lstm, self).__init__()
        # nn.
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers, batch_first=True)  # 使用两层lstm
        self.classifier = nn.Linear(hidden_feature, num_class)  # 将最后一个的rnn使用全连接的到最后的输出结果

        # self.inorm = torch.nn.InstanceNorm2d(num_features=1)

    def forward(self, x):
        # x = self.inorm(x.unsqueeze(1)).squeeze()
        # x = x.permute(1, 0, 2)
        # print(x.size())
        out, _ = self.rnn(x)  # 使用默认的隐藏状态，得到的out是（28， batch， hidden_feature）
        out = out[:, -1, :]
        out = self.classifier(out)
        return out



if __name__ == '__main__':
    model = BuildAlexNet(model_type='pre', n_output=2)
    # print(summary(model, (3, 224, 224)))
