# 按照DIFEX方法性能不高，只保留CORAL loss的时候，性能才能达到90%以上
import argparse
import torch
import time
import logging
from logging.handlers import RotatingFileHandler
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from confusion_matrix import confusion
from Network1 import my_resnet
from mmd_loss import mmd_loss
import numpy as np
from data_loader import read_train_data, read_test_data
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import random
import os
import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
timestape = datetime.datetime.now().strftime('%Y%m%d%H%M')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置pytorch内置hash函数种子，保证不同运行环境下字典等数据结构的哈希结果一致
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False禁用

def parse_args():
    parser = argparse.ArgumentParser(description="CORAL")
    parser.add_argument('--lam', type=float, default=0.05, help='lam')
    parser.add_argument('--beta', type=float, default=0.01, help='beta')
    return parser.parse_args()

def setup_logger(log_dir,filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'{filename}.log')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 10, backupCount=3)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

class Config:
    def __init__(
            self,
            batch_size: int = 32,
            test_batch_size: int = 32,
            epochs: int = 100,
            lr: float = 0.001,
            save_path: str = 'model_weight/model.pth',
            device_num: int = 0,
            rand_num: int = 30,
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path
        self.device_num = device_num
        self.rand_num = rand_num
        self.train_lossses = []
        self.val_accuracies = []

def coral(x, y):
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)
    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()
    return mean_diff + cova_diff

def train(model, train_dataloader, optimizer, epoch, writer, device_num, logger):
    model.train()
    tmodel = torch.load('model_weight/pretrain.pth')
    tmodel.eval()
    device = torch.device("cuda:" + str(device_num))
    correct = 0
    conf = parse_args()
    lam = conf.lam

    for data_nnl in train_dataloader:
        data, target, index = data_nnl
        target = target.squeeze().long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optimizer.zero_grad()
        features, output = model(data)
        classifier_output = F.log_softmax(output, dim=1)
        classifier_loss_batch = loss(classifier_output, target)
        with torch.no_grad():
            real_part = data[:, 0, :]
            imag_part = data[:, 1, :]
            complex_data = torch.complex(real_part, imag_part)
            data1 = torch.angle(torch.fft.fft(complex_data, dim=(1)))
            data1 = data1.unsqueeze(1)
            tfeatures, toutput = tmodel(data1)  # 这里不计算梯度
        loss2 = F.mse_loss(features[:, :256], tfeatures)*lam   # 主干网络提取的域内不变特征
        #loss3 = F.mse_loss(features[:, :256], features[:, 256:])*0.001
        loss4 = 0
        B = 32
        for i in range(10 - 1):
            for j in range(i + 1, 10):
                loss4 += coral(features[i * B:(i + 1) * B, 256:], features[j * B:(j + 1) * B, 256:])
        loss4 = beta * (loss4 * 2) / (10 * 9)
        loss1 = classifier_loss_batch
        total_loss = (loss1 + loss2 + loss4 ) / (1 + beta + lam)
        #total_loss = (loss1 + loss2 ) / (1 + lam)#只有F1
        #total_loss = (loss1 + loss4 ) / (1 + beta)#只有F2
        total_loss.backward()
        optimizer.step()
        total_loss += total_loss.item()
        pred = classifier_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    mean_loss = total_loss / len(train_dataloader)  # len(t1)表示t1数据集的批次数量
    mean_accuracy = 100 * correct / len(train_dataloader.dataset)  # 所有样本中预测正确的数量除以总的样本个数
    logger.info(
        'Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            mean_loss,
            correct,
            len(train_dataloader.dataset),
            mean_accuracy)
    )  # 打印轮数、分类器损失和准确率信息
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', mean_loss, epoch)  # 用于训练准确率和分类器损失的可视化
    return mean_loss, mean_accuracy

def evaluate(model, loss, val_dataloader, epoch, writer, device_num, logger):
    model.eval()
    val_loss = 0
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target, index in val_dataloader:
            target = target.squeeze().long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            features, output = model(data)
            output = F.log_softmax(output, dim=1)
            val_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_dataloader)
    mean_accuracy = 100 * correct / len(val_dataloader.dataset)
    fmt = '\nValidation set: loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    logger.info(
        fmt.format(
            val_loss,
            correct,
            len(val_dataloader.dataset),
            mean_accuracy,
        )
    )
    accuracy = 100.0 * correct / len(val_dataloader.dataset)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)

    return val_loss, mean_accuracy


def test(model, test_dataloader, logger):
    model.eval()
    correct = 0
    total_max_values_correct = 0  # 累加正确分类样本的最大概率值
    total_correct_samples = 0  # 计数正确分类的样本数量
    total_max_values = 0
    total_samples = 0
    total_max_values_wrong = 0  # 累加错误分类样本的最大概率值
    total_wrong_samples = 0  # 计数错误分类的样本数量
    features_list = []
    labels_list = []
    target_pred = []
    target_real = []
    probabilities_list = []  # 用于存储所有概率
    t = 0
    i = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            start_time = time.time()
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            features, output = model(data)
            out = F.log_softmax(output, dim=1)
            max_values, max_indices = torch.max(out, dim=1)
            total_max_values += max_values.sum().item()  # 累加每个batch的max_values总和
            total_samples += data.size(0)  # 累加样本数量

            pred = output.argmax(dim=1, keepdim=True)  # 预测结果
            correct_mask = pred.eq(target.view_as(pred))  # 创建一个正确分类的掩码
            correct += correct_mask.sum().item()

            max_values_correct = max_values[correct_mask.squeeze()]  # 选择正确分类的样本的最大概率值
            total_max_values_correct += max_values_correct.sum().item()  # 累加这些概率值
            total_correct_samples += correct_mask.sum().item()  # 累加正确分类的样本数

            # 错误分类的处理
            incorrect_mask = ~correct_mask.squeeze()  # 错误分类的掩码
            max_values_wrong = max_values[incorrect_mask]  # 选择错误分类的样本的最大概率值
            total_max_values_wrong += max_values_wrong.sum().item()  # 累加这些概率值
            total_wrong_samples += incorrect_mask.sum().item()  # 累加错误分类的样本数

            end_time = time.time()
            elapsed_time = end_time - start_time
            t += elapsed_time
            i += 1
            features_list.append(features.cpu().numpy())
            labels_list.append(target.cpu().numpy())
            target_pred.extend(pred.view(-1).tolist())
            target_real.extend(target.view(-1).tolist())
            probabilities_list.append(out.cpu().numpy())

        t /= i
        #logger.info("Average processing time per batch:", t)
        target_pred = np.array(target_pred)
        target_real = np.array(target_real)
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    accuracy = correct / len(test_dataloader.dataset)
    fmt = 'Accuracy: {}/{} ({:0f}%)\n'
    logger.info(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            accuracy,
        )
    )
    #logger.info(correct / len(test_dataloader.dataset))
    #logger.info("Accuracy:", accuracy)
    return target_pred, target_real, all_labels, all_features

def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, writer, save_path,
                       device_num, logger):
    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []
    current_max_val_accuracy = 1  # 初始化当前最小的测试损失为一个较大的值，判断模型是否有改进
    time_start1 = time.time()
    for epoch in range(1, epochs + 1):
        time_start = time.time()
        train_loss, train_acc = train(model, train_dataloader, optimizer, epoch, writer, device_num, logger)
        train_losses.append(train_loss)
        train_accies.append(train_acc)
        val_loss, val_accuracy = evaluate(model, loss_function, val_dataloader, epoch, writer, device_num, logger)
        val_losses.append(val_loss)
        val_accies.append(val_accuracy)
        if val_accuracy > current_max_val_accuracy:
            logger.info("The validation accuracy is increased from {} to {}, new model weight is saved.".format(
                current_max_val_accuracy, val_accuracy))
            current_max_val_accuracy = val_accuracy
            torch.save(model, save_path)
        else:
            logger.info("The validation loss is not decreased.")
        time_end = time.time()
        time_sum = time_end - time_start
        logger.info("time for each epoch is: %s" % time_sum)
        logger.info("------------------------------------------------")
        torch.cuda.empty_cache()  # 每轮训练结束清空未使用的显存
    time_end1 = time.time()
    Ave_epoch_time = (time_end1 - time_start1) / epochs
    logger.info("Avgtime for each epoch is: %s" % Ave_epoch_time)
    return train_losses, train_accies, val_losses, val_accies


class MultiFileBatchSampler(BatchSampler):
    def __init__(self, file_indices, batch_size_per_file):
        self.file_indices = file_indices  # 每个文件的数据索引
        self.batch_size_per_file = batch_size_per_file  # 每个文件的batch size
        self.num_files = len(file_indices)

    def __iter__(self):
        # 计算每个文件的样本总数
        num_samples_per_file = len(self.file_indices[0])
        num_batches = num_samples_per_file // self.batch_size_per_file

        for batch_idx in range(num_batches):
            batch = []
            for file_index in self.file_indices:
                # 从每个文件按顺序取出 batch_size_per_file 个数据
                start_idx = batch_idx * self.batch_size_per_file
                end_idx = start_idx + self.batch_size_per_file
                batch.extend(file_index[start_idx:end_idx])
            yield batch

    def __len__(self):
        num_samples_per_file = len(self.file_indices[0])
        return (num_samples_per_file // self.batch_size_per_file) * self.num_files


if __name__ == '__main__':
    conf = Config()  # 创建一个配置对象，存储各种配置参数
    writer = SummaryWriter("logs")

    device = torch.device("cuda:" + str(conf.device_num))
    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train_all, Y_train_all, L_train_all = [], [], []
    X_val_all, Y_val_all, L_val_all = [], [], []

    file_paths = [f'D:/work/data/paper6data/day1equalized/IQ/200/Rx{i}.npy' for i in range(1, 11)]
    for i, file_path in enumerate(file_paths):
        X_train, X_val, Y_train, Y_val, L_train, L_val = read_train_data(file_path, file_label=i)
        X_train_all.append(X_train)
        Y_train_all.append(Y_train)
        L_train_all.append(L_train)

        X_val_all.append(X_val)
        Y_val_all.append(Y_val)
        L_val_all.append(L_val)

    X_train_all = np.concatenate(X_train_all, axis=0)
    Y_train_all = np.concatenate(Y_train_all, axis=0)
    L_train_all = np.concatenate(L_train_all, axis=0)

    X_val_all = np.concatenate(X_val_all, axis=0)
    Y_val_all = np.concatenate(Y_val_all, axis=0)
    L_val_all = np.concatenate(L_val_all, axis=0)

    # 将数据转换为PyTorch张量，包含：数据、标签和文件标签
    train_dataset = TensorDataset(torch.Tensor(X_train_all), torch.Tensor(Y_train_all), torch.Tensor(L_train_all))
    val_dataset = TensorDataset(torch.Tensor(X_val_all), torch.Tensor(Y_val_all), torch.Tensor(L_val_all))

    file_indices = []
    file_size = len(X_train_all) // 10 # 假设每个文件的数据量相同
    for i in range(10):
        file_indices.append(list(range(i * file_size, (i + 1) * file_size)))
    batch_size_per_file = conf.batch_size  # 每个文件取32个数据
    batch_sampler = MultiFileBatchSampler(file_indices, batch_size_per_file)

    # train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)

    conf = parse_args()
    lam = conf.lam
    beta = conf.beta

    save_dir = f"tsne/"
    #ManyRx_111_1218是Mask后的，ManyRx_111_12181是原始的
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f'lam={conf.lam}_{conf.beta}_{timestape}'
    logger = setup_logger(save_dir, file_name)
    logger.info("Training session starts")
    logger.info(f"Configuration: {conf}")
    modelweightfile = os.path.join(save_dir, f'{file_name}.pth')


    model = my_resnet()
    logger.info(model)

    if torch.cuda.is_available():
        model = model.to(device)
    loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss = loss.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    save_path: str = 'model_weight/model.pth'
    device_num = 0
    time_start = time.time()

    time_end = time.time()
    time_sum = time_end - time_start
    logger.info("total training time is: %s" % time_sum)

    X_test, Y_test, = read_test_data()
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    data = torch.tensor(X_test)
    test_dataloader = DataLoader(test_dataset)
    model = torch.load('model_weight/model.pth')  # Raw是ResNet34的
    pred, real, labels, features = test(model, test_dataloader, logger)
    print(features.shape)
    #num_features = features.shape[1] * features.shape[2]
    num_features = features.shape[1]
    features_reshaped = features.reshape(-1, num_features)
    labels = labels.flatten()
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features_reshaped)
    plt.rc('font', family='Times New Roman')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.labelsize'] = 16  # 设置 x 轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 16  # 设置 y 轴刻度标签字体大小

    #colors = ['#427AB2', '#F09148', '#E56F5E', '#B395BD', '#C59D94', '#7DAEE0']
    #colors = ['#96C3D8', '#67A59B', '#A5D38F', '#f0d89a', '#F5B375', '#F19294']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    cmap = ListedColormap(colors)
    # 绘制颜色条
    # 绘制 t-SNE 散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap)
    # 添加颜色条
    cbar = fig.colorbar(scatter, orientation='vertical', ticks=np.arange(6))
    cbar.set_ticklabels([f'TX{i + 1}' for i in range(6)])
    cbar.ax.tick_params(labelsize=16)
    plt.savefig("D:/work/after.pdf", format='pdf', bbox_inches='tight')
    plt.show()





