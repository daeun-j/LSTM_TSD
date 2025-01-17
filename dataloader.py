import torch.utils.data as torch_data
import numpy as np
import pandas as pd
import torch
from scipy import stats
from numpy.fft import fft, fftfreq
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from numpy import inf
from sklearn.preprocessing import OneHotEncoder
import scipy
scaler = StandardScaler()
encoder = OneHotEncoder(sparse=False)

class Dataset(torch_data.Dataset):  # Inter
    # pre procession
    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        #         x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        #print("x_end_idx", x_end_idx)
        labels = [0, 1, 2]
        remove_ids = []
        for i in range(len(x_end_idx)):
            hi = x_end_idx[i]
            lo = hi - self.window_size
            target_data = self.target[lo: hi]
            y = target_data.mean()
            if y not in labels:
                remove_ids.append(i)

        del x_end_idx[remove_ids[1]]
        del x_end_idx[remove_ids[0]]
        return x_end_idx

    def targets(self, index):
        print("index", index)
        print("x_end_idx[index]", self.x_end_idx[index])
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        target_data = self.target[lo: hi]
        y = target_data.mean()
        return y

    def __init__(self, df, window_size, fft_num, stat, MERGE):
        self.window_size = window_size
        self.interval = window_size
        self.fft_num = fft_num
        self.stat = stat
        self.MERGE = MERGE
        self.horizon = 1
        self.data_time = df[:, 0]
        self.data_length = df[:, 1]
        self.target = df[:, 2]
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values


    def __getitem__(self, index):

        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data_time = self.data_time[lo: hi].reshape(-1, 1)
        train_data_time = scaler.fit_transform(train_data_time)
        train_data_length = self.data_length[lo: hi].reshape(-1, 1)
        target_data = self.target[lo: hi]
        y = target_data.mean()
        labels = [0, 1, 2]
        if y in labels:
            # x = np.concatenate((train_data_time, train_data_length), axis=None).reshape(-1, 1)
            meta0 = single2meta(train_data_time, self.fft_num, self.stat)
            meta1 = single2meta(train_data_length, self.fft_num, self.stat)
            meta2 = singles2intermeta(train_data_time.reshape(-1), train_data_length.reshape(-1))
            #x = x.reshape(-1)

            # if self.MERGE == 0:  # 머지 안함 [x]
            #     x = torch.from_numpy(x).type(torch.float)
            # elif self.MERGE == 7:  # Time만 [time]
            #     x = torch.from_numpy(train_data_time).type(torch.float)
            # elif self.MERGE == 9:  # len만 [length]
            #     x = torch.from_numpy(train_data_length).type(torch.float)
            #

            # elif self.MERGE == 1:  # meta0, meta1 만 머지 [x:meta0:meta1]
            #     x = np.concatenate((x, meta0, meta1), axis=None)
            #     x = torch.from_numpy(x).type(torch.float)
            #
            # elif self.MERGE == 2:  # meta2만 머지함.[x:meta2]
            #     x = np.concatenate((x, meta2), axis=None)
            #     x = torch.from_numpy(x).type(torch.float)
            # elif self.MERGE == 3:  # 모두 머지함.[x:meta0:meta1:meta2]
            #     x = np.concatenate((x, meta0, meta1, meta2), axis=None)
            #     x = torch.from_numpy(x).type(torch.float)


            if self.MERGE == 4:  # 메타만 머지함.[meta1:meta2]
                x = np.concatenate((meta0, meta1), axis=None)
                x = torch.from_numpy(x).type(torch.float)
            elif self.MERGE == 5:  # 메타만 머지함.[meta0:meta1:meta2]
                x = np.concatenate((meta0, meta1, meta2), axis=None)
                x = torch.from_numpy(x).type(torch.float)
            elif self.MERGE == 6:  # 메타만 머지함.[meta2]
                x = np.concatenate(meta2, axis=None)
                x = torch.from_numpy(x).type(torch.float)

            # elif self.MERGE == 8:  # Time+meta [time, meta0]
            #     x = np.concatenate((train_data_time, meta0), axis=None)
            #     x = torch.from_numpy(x).type(torch.float)
            #
            # elif self.MERGE == 10:  # len+meta [length, meta1]
            #     x = np.concatenate((train_data_length, meta1), axis=None)
            #     x = torch.from_numpy(x).type(torch.float)
            # elif self.MERGE == 11:  # meta
            #     x = np.concatenate(meta0, axis=None)
            #     x = torch.from_numpy(x).type(torch.float)
            # elif self.MERGE == 12:  # meta1
            #     x = np.concatenate(meta1, axis=None)
            #     x = torch.from_numpy(x).type(torch.float)

            x = x.reshape(-1)
            x = torch.unsqueeze(x, 1)
            x[x == inf] = 100000000

            # MSEloss
            # print("1" ,target_data.mean(), type(target_data.mean()))
            # target_data = encoder.fit_transform(target_data.mean().reshape(-1, 1))
            # print("2" ,target_data, type(target_data))
            # y = torch.from_numpy(target_data).type(torch.float).mean()
            # print("3" ,y, y.size())
            # y = encoder.fit_transform(y.view(-1, 1))

            # cross entropy
            y = torch.from_numpy(target_data).type(torch.float).mean()
        return x, y


class Dataset_raw(torch_data.Dataset):  # Inter
    # pre procession
    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        self.window_size=1
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

    def targets(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        target_data = self.target[lo: hi]
        y = target_data.mean()
        return y

    def __init__(self, df, MERGE):

        self.horizon = 1
        self.interval = 1

        self.df_length = len(df)
        self.MERGE = MERGE
        self.x_end_idx = self.get_x_end_idx()
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_time = df[:, 0]
        self.data_length = df[:, 1]
        self.target = df[:, 2]

    def __getitem__(self, index):

        hi = self.x_end_idx[index]
        lo = hi-1
        train_data_time = self.data_time[lo: hi].reshape(-1, 1)
        train_data_length = self.data_length[lo: hi].reshape(-1, 1)
        target_data = self.target[lo: hi]
        x = np.concatenate((train_data_time, train_data_length), axis=None).reshape(-1, 1)
        x = x.reshape(-1)

        if self.MERGE == 0:  # 머지 안함 [x]
            x = torch.from_numpy(x).type(torch.float)
        elif self.MERGE == 7:  # Time만 [time]
            x = torch.from_numpy(train_data_time).type(torch.float)
        elif self.MERGE == 9:  # len만 [length]
            x = torch.from_numpy(train_data_length).type(torch.float)

        x = x.reshape(-1)
        x = torch.unsqueeze(x, 1)

        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

# def single2meta(x, FFT_NUM, STAT):
#     meta_vector = []
#     if STAT == 1:
#         stat_features = [stats.skew(x, nan_policy="omit"), stats.kurtosis(x), stats.sem(x)]
#         meta_vector.extend(stat_features)
#     elif STAT != 1:
#         stat_features = []
#         meta_vector.extend(stat_features)
#
#     if FFT_NUM != 0:
#         freqs = fftfreq(x.shape[-1])
#         print("freqs", freqs)
#         mask = freqs > 0  # 한 파장당 지점 개수
#         fft_vales = np.fft.fft(x) / len(x)
#         print("fft_vales", fft_vales)
#         fft_norm = abs(fft_vales)
#         # fft_norm = fft_vales * (1.0 // x.shape[-1])  # FFT 계산된 결과를 정규화
#         fft_theo = 2.0 * abs(fft_norm)  # 푸리에 계수 계산
#         if sum(np.asarray(mask) * 1) == 0:
#             fft_len = 0
#         else:
#             fft_theo = fft_theo[mask]
#             fft_theo[::-1].sort()
#             fft = fft_theo[0:FFT_NUM].reshape(-1, )
#             fft_len = len(fft)
#             if fft_len < FFT_NUM:
#                 fft = np.append(fft, np.ones(FFT_NUM - len(fft)) * (1.e-5))
#             fft.tolist()
#             meta_vector.extend(fft)
#             print(fft)
#     elif FFT_NUM == 0:
#         fft = []
#         meta_vector.extend(fft)
#
#     meta = np.array(meta_vector)
#     meta.reshape(-1)
#     np.nan_to_num(meta, copy=False)
#
#     return meta

def single2meta(x, FFT_NUM, STAT):
    meta_vector = []
    if STAT == 1:
        stat_features = [stats.skew(x, nan_policy="omit"), stats.kurtosis(x), stats.sem(x)]
        meta_vector.extend(stat_features)
    elif STAT != 1:
        stat_features = []
        meta_vector.extend(stat_features)

    fft_vales = np.fft.fft(x) / len(x)
    real_fft = fft_vales.real
    sort_fft = np.sort(real_fft.flatten(), axis= -1)
    fft = sort_fft[-FFT_NUM:]

    meta = np.array(meta_vector)
    meta = np.append(meta, fft)
    meta.reshape(-1)
    np.nan_to_num(meta, copy=False)
    return meta


def singles2intermeta(x1, x2):
    intermeta_vector = [np.dot(x1, x2),
                        np.correlate(x1, x2)[0],
                        scipy.stats.kendalltau(x1, x2)[0]]
    intermeta_vector = np.array(intermeta_vector)
    intermeta_vector.reshape(-1)
    np.nan_to_num(intermeta_vector, copy=False)

    return intermeta_vector.reshape(-1, 1)




if __name__ == '__main__':
    fft_num = 3
    stat = 1
    MERGE = 9

    x = np.random.rand(30)
    y = np.random.rand(30)
    meta = single2meta(x, fft_num, stat)
    # print(meta)
    #meta = singles2intermeta(x, y)
    # print(meta)

    df = pd.DataFrame()

    df = pd.read_csv("dataset/Telegram_1hour_7.csv")
    df.insert(2, "label", int(0))
    df_0 = df[["Time", "Length", "label"]].to_numpy()

    df_set = Dataset_raw(df_0, MERGE=MERGE)

    train_dataset = DataLoader(dataset=df_set, batch_size=1, drop_last=False, shuffle=True, num_workers=0)

    for x, y in train_dataset:
        print("로더 y:", y, y.size())
        print("로더 x:", x, x.size()[1])
        continue
    #