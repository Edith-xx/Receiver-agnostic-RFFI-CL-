import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch
from numpy.random import standard_normal, uniform, randint
def awgn(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    print(x.shape)
    #snr_range = np.arange(-5, 20)
    #snr = uniform(snr_range[0], snr_range[-1], data_ch0.shape[0])
    #data_ch0 = awgn(data_ch0, snr)
    snr = 10
    snr = 10 ** (snr / 10.0)
    batch_size, num_features, seq_len = x.shape
    x_awgn = torch.zeros_like(x)
    x_real = x[:, 0, :]
    xpower_real = torch.sum(x_real ** 2, dim=1) / seq_len
    npower_real = xpower_real / snr
    noise_real = torch.randn(batch_size, seq_len) * torch.sqrt(npower_real.view(batch_size, 1))
    x_awgn[:, 0, :] = x_real + noise_real
    x_imag = x[:, 1, :]
    xpower_imag = torch.sum(x_imag ** 2, dim=1) / seq_len
    npower_imag = xpower_imag / snr
    noise_imag = torch.randn(batch_size, seq_len) * torch.sqrt(
        npower_imag.view(batch_size, 1))
    x_awgn[:, 1, :] = x_imag + noise_imag
    return x_awgn
def normalization(data):
    data = torch.tensor(data)
    magnitude = torch.sqrt(data[:, 0, :] ** 2 + data[:, 1, :] ** 2)  # [batch, N]
    max_magnitude = magnitude.max(dim=1, keepdim=True)[0]  # [batch, 1]
    max_magnitude = max_magnitude + 1e-8
    data[:, 0, :] = data[:, 0, :] / max_magnitude
    data[:, 1, :] = data[:, 1, :] / max_magnitude
    return data
def apply_random_mask(data, mask_ratio=0.2):
    num_samples, num_channels, num_features = data.shape
    mask = np.random.rand(num_samples, num_channels, num_features) > mask_ratio
    masked_data = data * mask
    return masked_data
class LoadDataset():
    def __init__(self):
        self.dataset_name = ' data'
        self.labelset_name = 'label'
    def load_stft_data(self, data_path):
        loaded = np.load(data_path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            data = loaded[loaded.files[0]]
        else:                           
            data = loaded
        data = data.transpose(0, 2, 1) 
        #data = awgn(data)
        #data = apply_random_mask(data, mask_ratio=0.10)
        return data
    def load_labels(self, label_path, dev_range):
        label = np.load(label_path)
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1
        sample_index_list = []
        for dev_idx in dev_range:
            num_pkt = np.count_nonzero(label == dev_idx)
            pkt_range = np.arange(0, num_pkt, dtype=int)
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
            print('Dev ' + str(dev_idx + 1) + ' have ' + str(num_pkt) + ' packets.')
        label = label[sample_index_list]
        return label
def read_train_data(file_path, file_label, dev_range=np.arange(0, 6, dtype=int),
                    label_path='D:/work/data/day1equalized/FFT/label_200.npy'):
    data_stft_all = []
    y_all = []
    label_all = [] 
    LoadDatasetObj = LoadDataset()
    data_ch0 = LoadDatasetObj.load_stft_data(file_path)
    y_ch0 = LoadDatasetObj.load_labels(label_path, dev_range)
    data_stft_all.append(data_ch0)
    y_all.append(y_ch0)
    file_labels = np.full((len(y_ch0),), file_label)
    label_all.append(file_labels)
    data_stft_all = np.concatenate(data_stft_all, axis=0)
    y_all = np.concatenate(y_all)
    label_all = np.concatenate(label_all)
    X_train, X_val, Y_train, Y_val, label_train, label_val = train_test_split(
        data_stft_all, y_all, label_all, test_size=0.2, random_state=32, shuffle=True)
    return X_train, X_val, Y_train, Y_val, label_train, label_val
def read_test_data(data_path='D:/work/data/day1equalized/IQ/200/Rx11_12.npy',
                   label_path='D:/work/data/day1equalized/FFT/label_400.npy',
                   dev_range=np.arange(0, 6, dtype=int),
                   test_size=0.2, random_state=42):
    data_stft_all = []
    y_all = []
    LoadDatasetObj = LoadDataset()
    data_ch0 = LoadDatasetObj.load_stft_data(data_path)
    y_ch0 = LoadDatasetObj.load_labels(label_path, dev_range)
    data_ch0_subset, _, y_ch0_subset, _ = train_test_split(
        data_ch0, y_ch0, test_size=(1 - test_size), random_state=random_state, stratify=y_ch0
    )
    data_stft_all.append(data_ch0)
    y_all.append(y_ch0)
    data_stft_all = np.concatenate(data_stft_all, axis=0)
    y_all = np.concatenate(y_all)
    return data_stft_all, y_all
def main():
    X_train, X_val, Y_train, Y_val, label_train, label_val = read_train_data()
    print(X_train.shape)




