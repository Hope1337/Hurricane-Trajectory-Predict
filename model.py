import torch
import torch.nn as nn
from load_data import Cus_Converter

class StormLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=5, output_size=2):
        super(StormLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.scaler = Cus_Converter()
    
    def load_pretrained(self, ckpt_path='checkpoint_epoch_5.pth'):
        self.load_state_dict(torch.load(ckpt_path))
    
    def predict_traj(self, clicked_point, n_point=5):
        with torch.no_grad():
            hidden = None
            out    = None
        
            for point in clicked_point:
                point     = self.scaler.point_scale(point)
                point     = torch.tensor(point).unsqueeze(0).unsqueeze(0).to("cuda")
                out, hidden = self.forward(point, hidden)
            
            predicted_points = []
            cur_point        = self.scaler.point_scale(clicked_point[-1])
        
            for i in range(n_point):
                predicted_points.append(self.scaler.point_convert(cur_point))
                out = out.squeeze(0).squeeze(0)
                out = [out[0].item(), out[1].item()]
                out = self.scaler.delta_convert(out)
                cur_point = self.scaler.point_convert(cur_point)
                cur_point[0] += out[0]
                cur_point[1] += out[1]
                cur_point = self.scaler.point_scale(cur_point)
                input = torch.tensor(cur_point).unsqueeze(0).unsqueeze(0).to("cuda")
                out, hidden = self.forward(input, hidden)
                
            return predicted_points

    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        #x.to("cuda")
        #hidden.to("cuda")
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden
    

