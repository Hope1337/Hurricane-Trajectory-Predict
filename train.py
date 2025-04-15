import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import StormLSTM
from load_data import get_data
from tqdm import tqdm  # Thư viện để hiển thị tiến độ
import sys 

# Hàm chuẩn hóa trajectories và offsets (đã có từ code trước)
def scale_trajectories_and_offsets(trajectories, offsets, long_scaler, lati_scaler, delta_lon_scaler, delta_lat_scaler):
    trajectories_scaled = []
    offsets_scaled = []
    
    for traj, offset in zip(trajectories, offsets):
        if len(traj) <= 1:
            continue
        long = np.array([point[0] for point in traj]).reshape(-1, 1)
        lati = np.array([point[1] for point in traj]).reshape(-1, 1)
        long_scaled = long_scaler.transform(long).flatten()
        lati_scaled = lati_scaler.transform(lati).flatten()
        traj_scaled = [[long_scaled[i], lati_scaled[i]] for i in range(len(traj))]
        trajectories_scaled.append(traj_scaled)

        try: 
            delta_lon = np.array([off[0] for off in offset]).reshape(-1, 1)
            delta_lat = np.array([off[1] for off in offset]).reshape(-1, 1)
            delta_lon_scaled = delta_lon_scaler.transform(delta_lon).flatten()
            delta_lat_scaled = delta_lat_scaler.transform(delta_lat).flatten()
            offset_scaled = [[delta_lon_scaled[i], delta_lat_scaled[i]] for i in range(len(offset))]
            offsets_scaled.append(offset_scaled)
        except:
            print(traj)
            print('#############################################3')
            print(offset)
            sys.exit()
    
    return trajectories_scaled, offsets_scaled

# Hàm huấn luyện mô hình với train/test split
def train_model():
    # Load dữ liệu

    print("loading data ...")
    trajectories, offsets, long_scaler, lati_scaler, delta_lon_scaler, delta_lat_scaler = get_data()
    
    # Chia dữ liệu thành train/test (80% train, 20% test)
    trajectories_train, trajectories_test, offsets_train, offsets_test = train_test_split(
        trajectories, offsets, test_size=0.2, random_state=42
    )
    # Chuẩn hóa dữ liệu train và test
    trajectories_train_scaled, offsets_train_scaled = scale_trajectories_and_offsets(
        trajectories_train, offsets_train, long_scaler, lati_scaler, delta_lon_scaler, delta_lat_scaler
    )
    trajectories_test_scaled, offsets_test_scaled = scale_trajectories_and_offsets(
        trajectories_test, offsets_test, long_scaler, lati_scaler, delta_lon_scaler, delta_lat_scaler
    )
    print('completed!')

    # Khởi tạo mô hình, loss function, và optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StormLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Huấn luyện mô hình trên tập train
    num_epochs = 20
    model.train()

    torch.autograd.set_detect_anomaly(True)
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0
        total_points = 0
        
        #for traj_idx, (trajectory, offset) in enumerate(zip(trajectories_train_scaled, offsets_train_scaled)):
        for traj_idx, (trajectory, offset) in enumerate(tqdm(zip(trajectories_train_scaled, offsets_train_scaled), 
                                                            total=len(trajectories_train_scaled), 
                                                            desc=f"Epoch {epoch+1}/{num_epochs}", 
                                                            leave=False)):
            trajectory = torch.tensor(trajectory, dtype=torch.float32).to(device)
            target = torch.tensor(offset, dtype=torch.float32).to(device)
            
            hidden = None

            traj_loss = 0.
            traj_points = 0
            
            for i in range(len(trajectory) - 1):
                point = trajectory[i:i+1, :].unsqueeze(0)
                ##############
                #print(point.shape)
                ##############
                target_point = target[i]
                
                output, hidden = model(point, hidden)
                output = output.squeeze(0).squeeze(0)
                #print(output.shape)
                #sys.exit()
                
                loss = criterion(output, target_point)
                traj_loss += loss
                traj_points += 1
                
                #loss.backward(retain_graph=True)
                #print(loss)
            
            traj_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += traj_loss.item()
            total_points += traj_points
                    
        avg_loss = total_loss / total_points if total_points > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

train_model()