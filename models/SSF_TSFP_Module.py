import torch.nn as nn
import torch


def conv_S_3x3(in_planes, out_planes, stride, padding):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=padding, bias=False)


def conv_T_3x3(in_planes, out_planes, stride, padding):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, padding=padding, bias=False)


def conv_S_9x9(in_planes, out_planes, stride, padding):
    # as is descriped, conv S is 1x7x7
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 9, 9), stride=stride, padding=padding, bias=False)


def conv_S_7x7(in_planes, out_planes, stride, padding):
    # as is descriped, conv S is 1x7x7
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 7, 7), stride=stride, padding=padding, bias=False)


def conv_S_5x5(in_planes, out_planes, stride, padding):
    # as is descriped, conv S is 1x5x5
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 5, 5), stride=stride, padding=padding, bias=False)

#v9
class Sidelayer_3x3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sidelayer_3x3x3, self).__init__()
        self.side_layer_3x3x3_conv1 = nn.Conv3d(in_channels, int(in_channels/2), kernel_size=1, bias=False)  # 1*1*1
        self.side_layer_3x3x3_conv_S_1 = conv_S_3x3(int(in_channels/2), int(in_channels/2), stride=1, padding=(0, 1, 1))  # 1*3*3卷积
        self.side_layer_3x3x3_conv_T_1 = conv_T_3x3(int(in_channels/2), int(in_channels/2), stride=1, padding=(1, 0, 0))  # 3*1*1卷积
        self.side_layer_3x3x3_conv2 = nn.Conv3d(int(in_channels/2), int(in_channels), kernel_size=1, bias=False)  # 1*1*1

        self.side_layer_3x3x3_conv3 = nn.Conv3d(int(in_channels), int(in_channels / 2), kernel_size=1, bias=False)  # 1*1*1
        self.side_layer_3x3x3_conv_S_2 = conv_S_3x3(int(in_channels / 2), int(in_channels / 2), stride=1, padding=(0, 1, 1))  # 1*3*3卷积
        self.side_layer_3x3x3_conv_T_2 = conv_T_3x3(int(in_channels / 2), int(in_channels / 2), stride=1, padding=(1, 0, 0))  # 3*1*1卷积
        self.side_layer_3x3x3_conv4 = nn.Conv3d(int(in_channels / 2), int(in_channels), kernel_size=1, bias=False)  # 1*1*1

        self.side_layer_3x3x3_conv5 = nn.Conv3d(int(in_channels), int(out_channels), kernel_size=1, bias=False)  # 1*1*1
        self.side_layer_3x3x3_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = []
        x = x[0]
        residual = x
        out = self.side_layer_3x3x3_conv1(x)
        out = self.side_layer_3x3x3_relu(out)
        out = self.side_layer_3x3x3_conv_S_1(out)
        out = self.side_layer_3x3x3_relu(out)
        out = self.side_layer_3x3x3_conv_T_1(out)
        out = self.side_layer_3x3x3_relu(out)
        out = self.side_layer_3x3x3_conv2(out)
        out = self.side_layer_3x3x3_relu(out+residual)


        residual = out
        out = self.side_layer_3x3x3_conv3(out)
        out = self.side_layer_3x3x3_relu(out)
        out = self.side_layer_3x3x3_conv_S_2(out)
        out = self.side_layer_3x3x3_relu(out)
        out = self.side_layer_3x3x3_conv_T_2(out)
        out = self.side_layer_3x3x3_relu(out)
        out = self.side_layer_3x3x3_conv4(out)
        out = self.side_layer_3x3x3_relu(out + residual)

        out = self.side_layer_3x3x3_conv5(out)
        out = self.side_layer_3x3x3_relu(out)
        output.append(out)
        return output


class Sidelayer_1x3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sidelayer_1x3x3, self).__init__()
        self.side_layer_3x3x3_conv_S = conv_S_3x3(in_channels, in_channels, stride=1, padding=(0, 1, 1))  # 1*3*3卷积
        self.side_layer_3x3x3_conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)  # 1*1*1
        self.side_layer_3x3x3_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = []
        x = x[0]
        out = self.side_layer_3x3x3_conv_S(x)
        out = self.side_layer_3x3x3_relu(out)
        out = self.side_layer_3x3x3_conv_S(out)
        out = self.side_layer_3x3x3_relu(out)
        out = self.side_layer_3x3x3_conv3(out)
        output.append(out)
        return output


class Fusion_Temp_For_Mask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion_Temp_For_Mask, self).__init__()
        self.fusion_1x1x1_1 = nn.Conv3d(in_channels, int(in_channels / 2), kernel_size=1, bias=False)  # 1*1*1
        self.fusion_relu = nn.ReLU(inplace=True)
        self.fusion_1x1x1_2 = nn.Conv3d(int(in_channels / 2), out_channels, kernel_size=1, bias=False)  # 1*1*1

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.fusion_1x1x1_1(x)
        out = self.fusion_relu(out)
        out = self.fusion_1x1x1_2(out)
        out = out.permute(0, 2, 1, 3, 4)
        return out


class TSFP_1(nn.Module):
    def __init__(self, in_planes=1, out_planes=1):
        super(TSFP_1, self).__init__()
        self.x_edge_layer1 = conv_S_9x9(in_planes, 16, stride=1, padding=(0, 4, 4))
        self.x_edge_layer2 = conv_S_7x7(16, 32, stride=1, padding=(0, 3, 3))
        self.x_edge_layer3 = conv_S_3x3(32, 64, stride=1, padding=(0, 1, 1))
        self.x_edge_layer4 = nn.Conv3d(64, 32, kernel_size=1, bias=False)  # 1*1*1
        self.x_edge_layer5 = conv_S_5x5(32, 16, stride=1, padding=(0, 2, 2))
        self.x_edge_layer6 = conv_S_3x3(16, 4, stride=(1, 2, 2), padding=(0, 1, 1))
        self.x_edge_layer7 = nn.Conv3d(4, out_planes, kernel_size=1, bias=False)
        self.x_edge_layer_relu = nn.ReLU(inplace=True)
        self.avgpooling_2 = nn.AvgPool3d(kernel_size=(1, 2, 2), padding=0, stride=(1, 2, 2))
        self.x_edge_layer_sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.avgpooling_2(x)
        out2 = self.x_edge_layer1(x)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer2(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer3(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer4(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer5(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer6(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer7(out2)
        out2 = self.x_edge_layer_sigmoid(out2)

        # deconv_feature = self.deconv(out2)
        final = torch.mul(out1, out2)

        return [final, out2]



class TSFP_2(nn.Module):
    def __init__(self, in_planes=1, out_planes=1):
        super(TSFP_2, self).__init__()
        self.x_edge_layer1 = conv_S_9x9(in_planes, 16, stride=1, padding=(0, 4, 4))
        self.x_edge_layer2 = conv_S_7x7(16, 32, stride=1, padding=(0, 3, 3))
        self.x_edge_layer3 = conv_S_3x3(32, 64, stride=1, padding=(0, 1, 1))
        self.x_edge_layer4 = nn.Conv3d(64, 32, kernel_size=1, bias=False)  # 1*1*1
        self.x_edge_layer5 = conv_S_5x5(32, 16, stride=1, padding=(0, 2, 2))
        self.x_edge_layer6 = conv_S_3x3(16, 4, stride=2, padding=(0, 1, 1))
        self.x_edge_layer7 = nn.Conv3d(4, out_planes, kernel_size=1, bias=False)
        self.x_edge_layer_relu = nn.ReLU(inplace=True)
        self.avgpooling = nn.AvgPool3d(kernel_size=2, padding=0, stride=2)
        self.x_edge_layer_sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.avgpooling(x)
        out2 = self.x_edge_layer1(x)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer2(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer3(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer4(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer5(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer6(out2)
        out2 = self.x_edge_layer_relu(out2)
        out2 = self.x_edge_layer7(out2)
        out2 = self.x_edge_layer_sigmoid(out2)
        final = torch.mul(out1, out2)

        return [final, out2]

