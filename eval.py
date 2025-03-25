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
from load_data import Cus_Converter

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

def eval():
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
    model = StormLSTM(input_size=2, hidden_size=50, num_layers=2, output_size=2).to(device)
    model.load_pretrained()
    model.eval()
    criterion = nn.MSELoss()
    total_test_loss = 0
    total_test_points = 0
    all_outputs = []
    all_targets = []
    
    long_loss = 0.
    lati_loss = 0.
    
    scaler = Cus_Converter()
    
    with torch.no_grad():
        #for traj_idx, (trajectory, offset) in enumerate(zip(trajectories_test_scaled, offsets_test_scaled)):
        for traj_idx, (trajectory, offset) in enumerate(tqdm(zip(trajectories_test_scaled, offsets_test_scaled), 
                                                             total=len(trajectories_test_scaled), 
                                                             desc="Evaluating on Test Set")):
            trajectory = torch.tensor(trajectory, dtype=torch.float32).to(device)
            target = torch.tensor(offset, dtype=torch.float32).to(device)
            
            hidden = None
            outputs = []
            
            for i in range(len(trajectory) - 1):
                point = trajectory[i:i+1, :].unsqueeze(0)
                output, hidden = model(point, hidden)
                outputs.append(output.squeeze(0).squeeze(0))
            
            outputs = torch.stack(outputs)
            all_outputs.append(outputs)
            all_targets.append(target)
            
            # Tính loss trên tập test
            for i in range(len(outputs)):
                #print(outputs[i].shape)
                #print(target[i].shape)
                #print()
                
                a = scaler.delta_convert([outputs[i][0].item(), outputs[i][1].item()])
                b = scaler.delta_convert([target[i][0].item(), target[i][1].item()])
                
                long_loss += abs(a[0] - b[0])
                lati_loss += abs(a[1] - b[1])
                
                loss = criterion(outputs[i], target[i])
                total_test_loss += loss.item()
                total_test_points += 1
    
    avg_test_loss = total_test_loss / total_test_points if total_test_points > 0 else 0
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test long Loss: {long_loss / total_test_points:.6f}")
    print(f"Test lati Loss: {lati_loss / total_test_points:.6f}")
    print()
    
    #from load_data import Cus_Converter
    #scaler = Cus_Converter()
    #print(scaler.delta_convert([avg_test_loss, avg_test_loss]))
    
    # Gộp tất cả các đầu ra và giá trị thực
    all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    
    # Đảo ngược chuẩn hóa
    delta_lon_outputs = all_outputs[:, 0].reshape(-1, 1)
    delta_lat_outputs = all_outputs[:, 1].reshape(-1, 1)
    delta_lon_targets = all_targets[:, 0].reshape(-1, 1)
    delta_lat_targets = all_targets[:, 1].reshape(-1, 1)
    
    delta_lon_outputs_unscaled = delta_lon_scaler.inverse_transform(delta_lon_outputs).flatten()
    delta_lat_outputs_unscaled = delta_lat_scaler.inverse_transform(delta_lat_outputs).flatten()
    delta_lon_targets_unscaled = delta_lon_scaler.inverse_transform(delta_lon_targets).flatten()
    delta_lat_targets_unscaled = delta_lat_scaler.inverse_transform(delta_lat_targets).flatten()
    
    all_outputs_unscaled = np.column_stack((delta_lon_outputs_unscaled, delta_lat_outputs_unscaled))
    all_targets_unscaled = np.column_stack((delta_lon_targets_unscaled, delta_lat_targets_unscaled))
    
    # Trực quan hóa kết quả trên tập test
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(all_targets_unscaled[:, 0], label='True ΔLON', color='blue')
    plt.plot(all_outputs_unscaled[:, 0], label='Predicted ΔLON', color='red', linestyle='--')
    plt.title('True vs Predicted ΔLON (Test Set)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(all_targets_unscaled[:, 1], label='True ΔLAT', color='blue')
    plt.plot(all_outputs_unscaled[:, 1], label='Predicted ΔLAT', color='red', linestyle='--')
    plt.title('True vs Predicted ΔLAT (Test Set)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    eval()