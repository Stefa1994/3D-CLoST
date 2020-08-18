from .imports import *
from .utils import normalize, denormalize

	
def create_dict(data, timestamps):

  # Funzione che crea un dizionario con matrice di inflow (_End) o outflow (_Start) per ogni timestamp.
  
  ny_dict = {}
  for index in range(len(data)):
    ny_dict[timestamps[index] + '_Inflow'] = data[index][0].tolist()
    ny_dict[timestamps[index] + '_Outflow'] = data[index][1].tolist()
  return ny_dict
  
def add_date_to_volume(city, timestamp, date_list, step, add_or_check = 'check', add_hour = True, add_half = False, add_day_before = False, add_week_before = False):
  
  if (city != 'NY') & (city != 'BJ'):
    return(print('You can insert NY or BJ as city'))
  elif (add_or_check != 'check') & (add_or_check != 'add'):
    return(print('You can only add hour to volume or check if you can insert hour'))
  elif (city == 'NY') & (add_half):
    return(print('You can not insert half an hour in New York Bike dataset'))
  else:
    if city == 'BJ':
      date_format = '%Y%m%d%H%M'
      last_day_minutes = [(24 * 60) - 30, 24 * 60, (24 * 60) + 30]
      last_week_minutes = [(24 * 60 * 7) - 30, 24 * 60 * 7, (24 * 60 * 7) + 30]
    else:
      date_format = '%Y%m%d%H'
      last_day_minutes = [(24 * 60) - 60, 24 * 60, (24 * 60) + 60]
      last_week_minutes = [(24 * 60 * 7) - 60, 24 * 60 * 7, (24 * 60 * 7) + 60]
    if add_or_check == 'check':
      operation = True
    else:
      operation = []
    for hour in range(1, step + 1):
      if add_or_check == 'check':
        if add_half:
          h = (dt.datetime.strptime(timestamp, date_format) - timedelta(hours = hour - 1, minutes = 30)).strftime(date_format)
          operation = operation & (h in date_list)
        if add_hour:
          h = (dt.datetime.strptime(timestamp, date_format) - timedelta(hours = hour)).strftime(date_format)
          operation = operation & (h in date_list)
      else:
        if add_half:
          h = (dt.datetime.strptime(timestamp, date_format) - timedelta(hours = hour - 1, minutes = 30)).strftime(date_format)
          operation.append(h)
        if add_hour:
          h = (dt.datetime.strptime(timestamp, date_format) - timedelta(hours = hour)).strftime(date_format)
          operation.append(h)
    if add_day_before:
      for minute in last_day_minutes:
        h = (dt.datetime.strptime(timestamp, date_format) - timedelta(minutes = minute)).strftime(date_format)
        if add_or_check == 'check':
          operation = operation & (h in date_list)
        else:
          operation.append(h)
    if add_week_before:
      for minute in last_week_minutes:
        h = (dt.datetime.strptime(timestamp, date_format) - timedelta(minutes = minute)).strftime(date_format)#last_week_minutes
        if add_or_check == 'check':
          operation = operation & (h in date_list)
        else:
          operation.append(h)
    if add_or_check == 'add':
      operation.reverse()

    return operation
	
	
	
def set_volume_date(city, date_list, step, add_hour = True, add_half = False, add_day_before = False, add_week_before = False):
  
  # Funzione che crea una lista contenente i timestamp con cui creare il volume.
  # Input: 
  #    - city: str. CittÃƒÂ  considerata. PuÃƒÂ² essere 'BJ' per il dataset taxi di Pechino o 'NY' per il dataset bici di New York
  #    - date_list: list. Lista di timestamp.
  #    - step: int. Numero di ore con cui creare il volume:
  #    - add_day_before: bool. Se True, aggiunge la stessa ora da predire del giorno precedente
  #    - add_week_before: bool. Se True, aggiunge la stessa ora da predire dello stesso giorno della settimana prima

  if (city != 'NY') & (city != 'BJ'):

    return(print('You can insert NY or BJ as city'))

  else:
    
    X_date, y_date = [], []
    if city == 'BJ':
      date_format = '%Y%m%d%H%M'
    elif city == 'NY':
      date_format = '%Y%m%d%H'
    for i in range(len(date_list)):
      try:
        timestamp = date_list[i + step]
        check_date = add_date_to_volume(city, timestamp, date_list, step, add_or_check = 'check', add_hour = add_hour, add_half = add_half, add_day_before = add_day_before, add_week_before = add_week_before)
        if not check_date:
            pass
        else:
          volume_date = add_date_to_volume(city, timestamp, date_list, step, add_or_check = 'add',  add_hour = add_hour, add_half = add_half, add_day_before = add_day_before, add_week_before = add_week_before)
          X_date.append(volume_date)
          y_date.append(timestamp)
      except:
        pass

    return X_date, y_date
	
	
	
def create_df_from_dict(city_dict):

    df_dict = {}
    index = 0
    for el in city_dict.keys():
      for i, val in enumerate(np.array(city_dict[el]).flatten()):
        df_dict[index] = {'Date_Type': el,
                          'Zone': i,
                          'Bike Number': val}
        index += 1

    df = pd.DataFrame.from_dict(df_dict, orient ='index')
    df['Date'] = df['Date_Type'].apply(lambda x: x.split('_')[0])
    df['Type'] = df['Date_Type'].apply(lambda x: x.split('_')[1])
    df.drop('Date_Type', inplace=True, axis=1)

    return df

def create_mask(city, city_dict):
  if city == 'NY':
    shape = (16, 8)
  else:
    shape = (32, 32)
  sum_inflow = np.zeros(shape = shape)
  sum_outflow = np.zeros(shape = shape)
  for i in city_dict.keys():
    if 'Inflow' in i:
      sum_inflow += city_dict[i]
    elif 'Outflow' in i:
      sum_outflow += city_dict[i]
  sum_outflow = np.array([0 if x == 0 else 1 for x in sum_outflow.flatten()]).reshape(shape)
  sum_inflow = np.array([0 if x == 0 else 1 for x in sum_inflow.flatten()]).reshape(shape)

  return np.array([sum_outflow, sum_inflow])
	
def create_normalized_volume(volume):
	shape = volume.shape
	volume = pd.Series(volume.flatten())
	vol_min = min(volume)
	vol_max = max(volume)
	volume = np.array(volume.apply(lambda x: normalize(x, vol_min, vol_max))).reshape(shape)
	return vol_min, vol_max, volume
  
def denormalize_volume(volume, min_x, max_x):
	volume = pd.Series(volume.flatten())
	return np.array(volume.apply(lambda x: denormalize(x, min_x, max_x))) 