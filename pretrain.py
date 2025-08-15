import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from confusion_matrix import confusion
from teacherNet import my_resnet
from data_loader import *
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
class Config:
    def __init__(
        self,
        batch_size: int = 32,
        test_batch_size: int = 8,
        epochs: int = 100,
        lr: float = 0.001,
        save_path: str = 'model_weight/pretrain.pth',
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
def train(model, train_dataloader, optimizer, epoch, writer, device_num):
    model.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    classifier_loss =0
    for data_nnl in train_dataloader:
        data, target = data_nnl
        real_part = data[:, 0, :]
        imag_part = data[:, 1, :]
        complex_data = torch.complex(real_part, imag_part)
        data = torch.angle(torch.fft.fft(complex_data, dim=(1)))
        data = data.unsqueeze(1)
        target = target.squeeze().long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optimizer.zero_grad()
        features, output = model(data)
        classifier_output = F.log_softmax(output, dim=1)
        classifier_loss_batch = loss(classifier_output, target)
        result_loss_batch = classifier_loss_batch
        result_loss_batch.backward()
        optimizer.step()
        classifier_loss += classifier_loss_batch.item()
        pred = classifier_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    mean_loss = classifier_loss / len(train_dataloader) 
    mean_accuracy = 100 * correct / len(train_dataloader.dataset)
    print(
        'Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            mean_loss,
            correct,
            len(train_dataloader.dataset),
            mean_accuracy)
        )                       
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', classifier_loss, epoch)
    return mean_loss, mean_accuracy
def evaluate(model, loss, val_dataloader, epoch, writer, device_num):
    model.eval()
    val_loss = 0
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    with torch.no_grad():
        for data, target in val_dataloader:
            real_part = data[:, 0, :]
            imag_part = data[:, 1, :]
            complex_data = torch.complex(real_part, imag_part)
            data = torch.angle(torch.fft.fft(complex_data, dim=(1)))
            data = data.unsqueeze(1)
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
    mean_accuracy = 100*correct/len(val_dataloader.dataset)
    fmt = '\nValidation set: loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
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
def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, writer, save_path, device_num):
    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []
    current_max_val_accuracy = 0                 
    time_start1 = time.time()
    for epoch in range(1, epochs + 1):
        time_start = time.time()
        train_loss, train_acc = train(model, train_dataloader, optimizer, epoch, writer, device_num)
        train_losses.append(train_loss)
        train_accies.append(train_acc)
        val_loss, val_accuracy = evaluate(model, loss_function, val_dataloader, epoch, writer, device_num)#val_accuracy是当前输入的验证批次数据的平均准确率
        val_losses.append(val_loss)
        val_accies.append(val_accuracy)

        if val_accuracy >= current_max_val_accuracy:
            print("The validation accuracy is increased from {} to {}, new model weight is saved.".format(
                current_max_val_accuracy, val_accuracy))
            current_max_val_accuracy = val_accuracy
            torch.save(model, save_path)
        else:
            print("The validation loss is not increased.")
        time_end = time.time()
        time_sum = time_end - time_start
        print("time for each epoch is: %s" % time_sum)
        print("------------------------------------------------")
        torch.cuda.empty_cache()
    time_end1 = time.time()
    Ave_epoch_time = (time_end1 - time_start1) / epochs
    print("Avgtime for each epoch is: %s" % Ave_epoch_time)
    return train_losses, train_accies, val_losses, val_accies
if __name__ == '__main__':
    conf = Config()                                                     
    writer = SummaryWriter("logs")
    device = torch.device("cuda:"+str(conf.device_num))
    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    run_for = 'Train'
    if run_for == 'Train':
        X_train_all, Y_train_all, L_train_all = [], [], []
        X_val_all, Y_val_all, L_val_all = [], [], []
        file_paths = [f'D:/work/data/paper6data/day1equalized/IQ/200/Rx{i}.npy' for i in range(1, 11)]
        for i, file_path in enumerate(file_paths):
            print(i)
            X_train, X_val, Y_train, Y_val, L_train, L_val = read_train_data(file_path, file_label=i)
            X_train_all.append(X_train)
            Y_train_all.append(Y_train)
            X_val_all.append(X_val)
            Y_val_all.append(Y_val)
        X_train_all = np.concatenate(X_train_all, axis=0)
        Y_train_all = np.concatenate(Y_train_all, axis=0)
        X_val_all = np.concatenate(X_val_all, axis=0)
        Y_val_all = np.concatenate(Y_val_all, axis=0)
        train_dataset = TensorDataset(torch.Tensor(X_train_all), torch.Tensor(Y_train_all))
        val_dataset = TensorDataset(torch.Tensor(X_val_all), torch.Tensor(Y_val_all))
        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        model = my_resnet()
        print(model)
        if torch.cuda.is_available():
            model = model.to(device)
        loss = nn.NLLLoss()
        if torch.cuda.is_available():
            loss = loss.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0)
        train_losses, train_accies, val_losses, val_accies = train_and_evaluate(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                       optimizer=optim, epochs=conf.epochs, writer=writer, save_path=conf.save_path,
                       device_num=conf.device_num)




