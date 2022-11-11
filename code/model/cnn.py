import torch
from torch import nn
import config


class FemnistCNN(nn.Module):
    """simple CNN model for FEMNIST

    Attributes:
        num_classes (int): number of classes
    """

    def __init__(self, num_classes=62):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2))
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, (5, 5), padding=(2, 2))
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2048, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        features = self.extract_features(x)

        output = self.fc2(features)

        return output

    def extract_features(self, x):
        features = x
        # N C H W
        batch_size = features.shape[0]  # batch size
        features = torch.reshape(features, (-1, 1, 28, 28))

        features = self.conv1(features)
        features = self.relu1(features)
        features = self.pool1(features)

        features = self.conv2(features)
        features = self.relu2(features)
        features = self.pool2(features)

        features = torch.reshape(features, (batch_size, -1))

        features = self.fc1(features)
        features = self.fc1_relu(features)

        return features


class CelebaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.num_classes = num_classes

        h_channel = 32
        self.conv_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, h_channel, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(h_channel),
                nn.MaxPool2d((2, 2)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(h_channel, h_channel, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(h_channel),
                nn.MaxPool2d((2, 2)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(h_channel, h_channel, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(h_channel),
                nn.MaxPool2d((2, 2), padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(h_channel, h_channel, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(h_channel),
                nn.MaxPool2d((2, 2), padding=1),
                nn.ReLU(inplace=True),
            )
        ])

        self.classifier = nn.Linear(1152, self.num_classes)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)

        return x

    def extract_features(self, x):
        for conv in self.conv_forwards:
            x = conv(x)
        x = x.flatten(1)
        return x


class C3D(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU(inplace=True)

        if pretrained:
            self.load_pretrained_weights(config.MODELS_UCF_PRETRAINED_PATH)
        else:
            self._init_weight()

    def forward(self, x):
        x = self.extract_features(x)
        logits = self.fc8(x)

        return logits

    def extract_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        return x

    def load_pretrained_weights(self, model_path):
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",
        }
        pretrained_dict = torch.load(model_path)
        print("Loaded pretrained weights from '{}'".format(model_path))

        self_dict = self.state_dict()
        for name in pretrained_dict:
            if name in corresp_name:
                self_dict[corresp_name[name]] = pretrained_dict[name]

        self.load_state_dict(self_dict)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_base_params(self):
        layers = [self.conv1, self.conv2, self.conv3a, self.conv3b, self.conv4a, self.conv4b,
                  self.conv5a, self.conv5b, self.fc6, self.fc7]
        for layer in layers:
            for param in layer.parameters():
                if param.requires_grad:
                    yield param

    def get_top_params(self):
        layers = [self.fc8]
        for layer in layers:
            for param in layer.parameters():
                if param.requires_grad:
                    yield param
