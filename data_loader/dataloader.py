import torch.utils.data as torch_data
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats
from numpy.fft import fft, fftfreq
import statsmodels.api as sm

def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data



class LSTMTSD_Dataset(torch_data.Dataset):
    # pre procession
    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1):
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.target = df[:, 2]
        self.data = df[:, 0:2]
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)

# df_set.__getitem__(0)[0][:, 0]
    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data = self.data[lo: hi]
        target_data = self.target[lo: hi]
        #print("train_data;", train_data)
        #print("train_data[:, 0];", train_data[:, 0])
        #print(train_data[:, 2])
        #print(train_data[:, 0:2])
        x = torch.from_numpy(train_data).type(torch.float)
        #print(x)
        meta = torch.from_numpy(x2meta(train_data[:, 0], 4, 2)).type(torch.float)
        #print("meta;", meta)
        y = torch.from_numpy(target_data).type(torch.float).mean()
        # if y==0:
        #     y= torch.Tensor([1, 0, 0])
        # elif y==1:
        #     y= torch.Tensor([0, 1, 0])
        # elif y==2:
        #     y= torch.Tensor([0, 0, 1])
        #meta_2 = torch.from_numpy(x2meta(x[:, 1], 4, 2)).type(torch.float)


        x = x.reshape(-1)
        x = torch.unsqueeze(x, 1)
        #print("x, x.size():", x, x.size())
        meta = torch.unsqueeze(meta, 1)

        return x, y, meta

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

def x2meta(x, FFT_NUM, ARIMA_LAGS):


    # statiatical : 4
    prob_vector = [stats.skew(x), stats.kurtosis(x),
                   stats.iqr(x), stats.sem(x)]
    prob_vector = np.array(prob_vector)
    prob_vector.reshape(-1)
    # FFT : 4
    freqs = fftfreq(x.shape[-1])  # 필요한 모든 진동수를 만든다.
    mask = freqs > 0  # 한 파장당 지점 개수
    fft_vales = np.fft.fft(x)
    fft_norm = fft_vales * (1.0 / x.shape[-1])  # FFT 계산된 결과를 정규화
    fft_theo = 2.0 * abs(fft_norm)  # 푸리에 계수 계산
    fft_theo = fft_theo[mask]
    fft_theo[::-1].sort()
    fft = fft_theo[0:FFT_NUM].reshape(-1,)
    if len(fft) < FFT_NUM:
        fft = np.append(fft, np.ones(FFT_NUM-len(fft))*(1.e-5))
    # # ARIMA : 2 * 2
    acf, _ = sm.tsa.acf(x, fft=False, alpha=0.05, nlags=ARIMA_LAGS)
    # pacf, _ = sm.tsa.pacf(x, alpha=0.05, nlags=ARIMA_LAGS)
    acf = acf[1:].reshape(-1)
    if len(acf)<ARIMA_LAGS:
        acf = np.append(acf, np.ones(ARIMA_LAGS-len(acf))*(1.e-5))
    # pacf = pacf[1:].reshape(-1)  # numpy.ndarray
    # TODO : meta = between, feature 마다 따로따로하는 것 생각하기.

    #meta = np.concatenate([prob_vector, fft, acf, pacf], axis=0)
    meta = np.concatenate([prob_vector, fft, acf], axis=0)
    np.nan_to_num(meta, copy=False)

    return meta

if __name__ == '__main__':
    ## 해당 경로에 있는 .csv 파일명 리스트 가져오기


    # path = '../dataset/'
    # file_list = os.listdir(path)
    # file_list_py = [file for file in file_list if file.endswith('.csv')] ## 파일명 끝이 .csv인 경우

    ## csv 파일들을 DataFrame으로 불러와서 concat

    df = pd.DataFrame()

    df = pd.read_csv("../dataset/Telegram_1hour_7.csv")
    df.insert(2, "label", int(0))
    df_0 = df[["Time", "Length", "label"]].to_numpy()

    df = pd.read_csv("../dataset/Zoom_1hour_5.csv")
    df.insert(2, "label", int(1))
    df_1 = df[["Time", "Length", "label"]].to_numpy()

    df = pd.read_csv("../dataset/YouTube_1hour_2.csv")
    df.insert(2, "label", int(2))
    df_2 = df[["Time", "Length", "label"]].to_numpy()

    df_set = np.vstack((df_0, df_1, df_2))
    # print(df_set.shape)
    # print(type(df_set))
    df_set = LSTMTSD_Dataset(df_set, window_size=5, horizon=1, normalize_method="z_score")
    #dataloader = DataLoader(df_set, batch_size=1, drop_last=False, shuffle=True, num_workers=0)
    train_dataset, val_dataset = torch.utils.data.random_split(df_set, [int(len(df_set)*0.8),len(df_set) - int(len(df_set)*0.8)])
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [int(len(val_dataset)*0.8),len(val_dataset) - int(len(val_dataset)*0.8)])

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, drop_last=False, shuffle=True, num_workers=0)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=1, drop_last=False, shuffle=True, num_workers=0)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, drop_last=False, shuffle=True, num_workers=0)

    print("train_dataset:", len(train_dataset))
    print("val_dataset:", len(val_dataset))
    print("test_dataset:", len(test_dataset))

    for x, y, meta in test_loader:
            print("로더 y:", y, y.size())
            print("로더 x:", x, x.size())
            print("로더 meta:", meta, meta.size())


