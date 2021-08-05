import numpy as np
import torch.utils.data
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, accuracy_score
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from torch.optim import lr_scheduler

def shedulers(optimizer, type):
    shedulers = dict({"StepLR": lr_scheduler.StepLR(optimizer, step_size=1, gamma= 0.99),
                  "MultiStepLR": lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,40], gamma= 0.1),
                  "ExponentialLR": lr_scheduler.ExponentialLR(optimizer, gamma= 0.99),
                  "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau(optimizer,threshold=1,patience=1,mode='min')})
    return shedulers[type]
def even_sampling(w,df_0, df_1, df_2): # 모든 클래스 instance 동
    min_len = min(len(df_0), len(df_1), len(df_2))
    min_len = min_len // w * w
    df_stack = np.vstack((df_0[:min_len], df_1[:min_len], df_2[:min_len]))
    return df_stack

def sampling(w,fix_num, df_0, df_1, df_2): #
    min_len = w * fix_num
    df_stack = np.vstack((df_0[:min_len], df_1[:min_len], df_2[:min_len]))
    return df_stack


def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)


def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)

def Acc(v, v_, axis=None):
    '''
    Accuracy
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, Accuracy on all elements of input.
    '''
    return (np.sum(((v==v_)*1))/v.shape[0]).astype(np.float64)



def evaluate(y, y_hat, by_step=False, by_node=False):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat), Acc(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0), Acc(y, y_hat)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2)), Acc(y, y_hat)
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1)), Acc(y, y_hat)



def evaluate_class_master(y, y_hat, by_step=False, by_node=False):
    confusion_matrix(y, y_hat, labels=[0, 1, 2])
    print(classification_report(y, y_hat))
    print("Accracy {} | macro recall {} micro recall {} | macro precision {}, micro precision {}".format(accuracy_score(y, y_hat),
                                                                                                         recall_score(y, y_hat, average='macro'),
                                                                                                         recall_score(y, y_hat, average='micro'),
                                                                                                     precision_score(y, y_hat, average='macro'),
                                                                                                         precision_score(y, y_hat, average='micro')))
    return accuracy_score(y, y_hat),recall_score(y, y_hat, average='macro'), recall_score(y, y_hat, average='micro'),precision_score(y, y_hat, average='macro'), precision_score(y, y_hat, average='micro')


# TEST reference
def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                forecast_result, a = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size, :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)

def validate(target, forecast, result_file=None):
    score = evaluate_class(target, forecast)
    print(f'Accracy  {score[0]:7.9} | macro precision {score[1]:7.9f} |  macro recall  {score[2]:7.9}')
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        forcasting_2d = forecast
        forcasting_2d_target = target

        np.savetxt(f'{result_file}/target.txt', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.txt', forcasting_2d, delimiter=",")
        # np.savetxt(f'{result_file}/predict_abs_error.txt',
        #            np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        # np.savetxt(f'{result_file}/predict_ape.txt',
        #            np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")
    return dict(Acc=score[0], mac_precision=score[1], mac_recall=score[2])

def evaluate_class(y, y_hat):
    # confusion_matrix(y, y_hat, labels=[0, 1])
    # confusion_matrix(y, y_hat, labels=[0, 1, 2])

    #print(classification_report(y, y_hat))
    #print("y, y_hat", y, y_hat)
    #print("Accracy {}  | macro precision {}| macro recall {}".format(accuracy_score(y, y_hat),precision_score(y, y_hat, average='macro'), recall_score(y, y_hat, average='macro'))
    return accuracy_score(y, y_hat), precision_score(y, y_hat, average='macro'), recall_score(y, y_hat, average='macro')


def anormal_dataset(dataset):
    df = pd.DataFrame()
    if dataset ==0:
        # todo : CIDDS 0
        df = pd.read_csv("dataset/Anormaly_Detection_data/CIDDS_001_external_week1.csv")
        # df["class"]
        # df["class"].unique()
        # freq = df.groupby(["class"]).count()
        # print(freq)
        normal = df[df["class"]=="normal"]
        normal.insert(0, "label", int(0))
        normal = normal[["Duration", "Packets", "label"]].to_numpy()

        anomaly = df[df["class"]=="suspicious"]
        anomaly.insert(0, "label", int(1))
        anomaly = anomaly[["Duration", "Packets", "label"]].to_numpy()


    elif dataset == 1:
        # todo : Android_malware 1
        df = pd.read_csv("dataset/Anormaly_Detection_data/net_android_malware.csv", sep=';')
        df.info()
        df["type"].unique()
        normal = df[df["type"]=="benign"]
        normal.insert(0, "label", int(0))
        normal = normal[["dns_query_times", "vulume_bytes", "label"]].to_numpy()

        anomaly = df[df["type"]=="malicious"]
        anomaly.insert(0, "label", int(1))
        anomaly = anomaly[["dns_query_times", "vulume_bytes", "label"]].to_numpy()


    elif dataset == 2:

        # todo : KDD_CUP99 2
        df = pd.read_csv("dataset/Anormaly_Detection_data/KDD_CUP99.csv")
        normal = df[df["attack"]=="normal"]
        normal.insert(0, "label", int(0))
        normal = normal[["Duration", "Src_bytes", "label"]].to_numpy()

        anomaly = df[df["attack"]!="normal"]
        anomaly.insert(0, "label", int(1))
        anomaly = anomaly[["Duration", "Src_bytes", "label"]].to_numpy()


    elif dataset == 3:
    #todo : virtual_linux 이상함
        df = pd.read_csv("dataset/Anormaly_Detection_data/virtual_linux_malware.csv")
        for i in list(df.columns):
            print(i, len(df["vm_truncate_count"].unique()), df[i].unique())

        df.info()
        df["classification"].unique()
        normal = df[df["classification"]=="malware"]
        normal.insert(0, "label", int(0))
        normal = normal[["utime", "vm_truncate_count", "label"]].to_numpy()

        anomaly = df[df["classification"]=="benign"]
        anomaly.insert(0, "label", int(1))
        anomaly = anomaly[["utime", "vm_truncate_count", "label"]].to_numpy()

    df_set = np.vstack((normal, anomaly))
    return df_set