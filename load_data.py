import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_trajectories(file = 'ibtracs.WP.list.v04r01.csv'):
    df   = pd.read_csv(file, low_memory=False)

    unique_sids = df['SID'].unique()[1:]

    long = df['LON'][1:].astype(float)
    lati = df['LAT'][1:].astype(float)

    ## Biểu đồ phân bố kinh độ
    #plt.subplot(1, 2, 1)
    #sns.histplot(long, bins=100, kde=True)
    #plt.title('Phân bố Kinh độ (Longitude)')
    #plt.xlabel('Kinh độ')
    #plt.ylabel('Số lượng')

    ## Biểu đồ phân bố vĩ độ
    #plt.subplot(1, 2, 2)
    #sns.histplot(lat, bins=100, kde=True)
    #plt.title('Phân bố Vĩ độ (Latitude)')
    #plt.xlabel('Vĩ độ')
    #plt.ylabel('Số lượng')

    #plt.tight_layout()
    #plt.show()

    trajectories = []

    for sid in unique_sids:
        mask = (df['SID'] == sid)[1:]

        tlong = long[mask]
        tlati = lati[mask]

        traj = [[x, y] for (x, y) in zip(tlong, tlati)] 

        trajectories.append(traj)
    

    return trajectories

def get_offset(trajectories):
    offsets = []

    for traj in trajectories:
        offset = []
        for i in range(1, len(traj)):
            offset.append([traj[i][0] - traj[i-1][0], traj[i][1] - traj[i-1][1]])  
        offsets.append(offset)

    return offsets

def get_scaler(trajectories):

    long = [tt[0] for t in trajectories for tt in t]
    lati = [tt[1] for t in trajectories for tt in t]

    long = np.array(long).reshape(-1, 1)
    lati = np.array(lati).reshape(-1, 1)

    long_scaler = MinMaxScaler()
    lati_scaler = MinMaxScaler()

    long_scaler.fit(long)
    lati_scaler.fit(lati)

    return long_scaler, lati_scaler

def get_data():
    trajectories = get_trajectories()
    offsets      = get_offset(trajectories)

    a, b = get_scaler(trajectories)
    c, d = get_scaler(offsets)

    return trajectories, offsets, a, b, c, d

class Cus_Converter():
    def __init__(self):
        trajectories, offsets, long_scaler, lati_scaler, delta_lon_scaler, delta_lat_scaler = get_data()
        self.long_scaler = long_scaler 
        self.lati_scaler = lati_scaler
        self.delta_lon_scaler = delta_lon_scaler
        self.delta_lat_scaler = delta_lat_scaler
        
    def point_scale(self, point):
        a = np.array(point[0]).reshape(-1, 1)
        b = np.array(point[1]).reshape(-1, 1)
        
        a = self.long_scaler.transform(a)
        b = self.lati_scaler.transform(b)

        return [float(a[0][0]), float(b[0][0])]
    
    def delta_scale(self, point):
        a = np.array(point[0]).reshape(-1, 1)
        b = np.array(point[1]).reshape(-1, 1)
        
        a = self.delta_lon_scaler.transform(a)
        b = self.delta_lat_scaler.transform(b)

        return [float(a[0][0]), float(b[0][0])]
        
    def point_convert(self, point): 
        a = np.array(point[0]).reshape(-1, 1)
        b = np.array(point[1]).reshape(-1, 1)
        
        a = self.long_scaler.inverse_transform(a)
        b = self.lati_scaler.inverse_transform(b)

        return [float(a[0][0]), float(b[0][0])]
    
    def delta_convert(self, point):
        a = np.array(point[0]).reshape(-1, 1)
        b = np.array(point[1]).reshape(-1, 1)
        
        a = self.delta_lon_scaler.inverse_transform(a)
        b = self.delta_lat_scaler.inverse_transform(b)

        return [float(a[0][0]), float(b[0][0])]

if __name__ == "__main__":
    trajectories = get_trajectories()
    offsets      = get_offset(trajectories)
    a, b, c, d   = get_scaler(trajectories)

    print(c.inverse_transform(np.array([[0.5]])))
        