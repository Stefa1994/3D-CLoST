from .imports import *

def normalize(X, min_x, max_x):
    X = 1. * (X - min_x) / (max_x - min_x)
    return X
  
def denormalize(X, min_x, max_x):
    X = 1. * X * (max_x - min_x) + min_x
    return X


def root_mean_squared_error(y_true, y_pred):
  
  # Keras RMSE  
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mean_absolute_percentage_error(y_true, y_pred):
    idx = y_true > 10
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100
    
    
    
def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps
    
    
def extract_data(city):
    if (city != 'NY') & (city != 'BJ'):
        return(print('You can insert NY or BJ as city'))
    else:
        path = '../data/raw/' + city
        if city == 'NY':
            print('Extracting data for NYCBike dataset..')
          #f = h5py.File('../data/raw/' + city + '/raw_data.h5', 'r')
            f = h5py.File('../data/raw/' + city + '/NYC14_M16x8_T60_NewEnd.h5', 'r')
            data = f['data'].value
            timestamps = f['date'].value
            f.close()
            np.save('../data/' + city + '/data.npy', data)
            np.save('../data/' + city + '/timestamps.npy', timestamps) 
            print('NYCBike dataset extracted')
        elif city == 'BJ':
            print('Extracting data for BJTaxi dataset..')
            all_data = []
            all_timestamps = []
            for year in range(13,17):
                #f = h5py.File('../data/raw/' + city + '/raw_data_' + str(year) + '.h5', 'r')
                f = h5py.File('../data/raw/' + city + '/BJ' + str(year) + '_M32x32_T30_InOut.h5', 'r')
                data = f['data'].value
                timestamps = f['date'].value
                f.close()
                np.save('../data/' + city + '/data_' + str(year) +'.npy', data)
                np.save('../data/' + city + '/timestamps_' + str(year) +'.npy', timestamps)
                all_data.append(data)
                all_timestamps.append(timestamps)
            np.save('../data/' + city + '/data.npy', data)
            np.save('../data/' + city + '/timestamps.npy', timestamps)
            print('BJTaxi dataset extracted')

      
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))