import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from model import StormLSTM
import torch


class StormMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Storm Trajectory Map")
        
        # Đọc dữ liệu
        self.df = pd.read_csv('ibtracs.WP.list.v04r01.csv')
        self.init_model()
        
        # Danh sách các điểm được click, các đường nối và các điểm dự đoán
        self.clicked_points = []
        self.red_lines = []
        self.predicted_points = []
        
        # Tạo figure với cartopy
        self.fig = plt.Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Thiết lập phạm vi bản đồ (Western Pacific region)
        self.ax.set_extent([100, 180, -10, 50], crs=ccrs.PlateCarree())
        
        # Thêm các feature cho bản đồ
        self.ax.add_feature(cfeature.LAND)
        self.ax.add_feature(cfeature.OCEAN)
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Tạo canvas để hiển thị trong tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        
        # Biến để hỗ trợ kéo thả và mode
        self.pressed = False
        self.x0 = None
        self.y0 = None
        self.mode = tk.StringVar(value="click")  # Mặc định là Click Mode
        
        # Bind events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        # Frame cho các nút
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM)
        
        # Thêm nút zoom
        zoom_in_btn = tk.Button(button_frame, text="Zoom In", command=self.zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=5)
        
        zoom_out_btn = tk.Button(button_frame, text="Zoom Out", command=self.zoom_out)
        zoom_out_btn.pack(side=tk.LEFT, padx=5)
        
        undo_btn = tk.Button(button_frame, text="Undo", command=self.undo)
        undo_btn.pack(side=tk.LEFT, padx=5)
        
        # Thêm radio buttons cho mode
        tk.Radiobutton(button_frame, text="Click Mode", variable=self.mode, 
                      value="click", command=self.update_cursor).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(button_frame, text="Pan Mode", variable=self.mode, 
                      value="pan", command=self.update_cursor).pack(side=tk.LEFT, padx=5)

    def update_cursor(self):
        # Thay đổi con trỏ chuột dựa trên mode
        cursor = "crosshair" if self.mode.get() == "click" else "hand2"
        self.canvas.get_tk_widget().config(cursor=cursor)

    def plot_storm_trajectories(self):
        storms = self.df.groupby('name')
        for name, group in storms:
            self.ax.plot(group['long'], group['lat'], 
                        color='blue', 
                        linewidth=1,
                        transform=ccrs.PlateCarree())
            first_point = group.iloc[0]
            self.ax.plot(first_point['long'], first_point['lat'],
                        'bo',
                        transform=ccrs.PlateCarree())

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        self.pressed = True
        self.x0 = event.xdata
        self.y0 = event.ydata
        
        if self.mode.get() == "click":
            self.add_point(event)

    def add_point(self, event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            # Thêm điểm đỏ
            point, = self.ax.plot(x, y, 'ro', transform=ccrs.PlateCarree())
            
            # Nối với điểm trước đó nếu có
            if self.clicked_points:
                last_x, last_y = self.clicked_points[-1][:2]
                line, = self.ax.plot([last_x, x], [last_y, y], 
                                   'r-', 
                                   transform=ccrs.PlateCarree())
                self.red_lines.append(line)
            
            self.clicked_points.append((x, y, point))
            self.predict_traj()
            self.canvas.draw()

    def on_motion(self, event):
        if not self.pressed or event.inaxes != self.ax or self.mode.get() != "pan":
            return
        if event.xdata is None or event.ydata is None:
            return
        
        dx = event.xdata - self.x0
        dy = event.ydata - self.y0
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
        self.ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
        
        self.canvas.draw()
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        self.pressed = False

    def undo(self):
        if self.clicked_points:
            # Xóa điểm cuối cùng
            last_point = self.clicked_points.pop()
            last_point[2].remove()  # Xóa marker
            
            # Xóa đường nối cuối cùng nếu có
            if self.red_lines:
                last_line = self.red_lines.pop()
                last_line.remove()
            
            # Xóa tất cả các điểm dự đoán
            for pred_point in self.predicted_points:
                pred_point.remove()
            self.predicted_points = []
            
            # Vẽ lại các điểm dự đoán nếu còn điểm click
            if self.clicked_points:
                self.predict_traj()
            
            self.canvas.draw()

    def zoom_in(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        new_xlim = [xlim[0] + (xlim[1] - xlim[0]) * 0.2, 
                   xlim[1] - (xlim[1] - xlim[0]) * 0.2]
        new_ylim = [ylim[0] + (ylim[1] - ylim[0]) * 0.2, 
                   ylim[1] - (ylim[1] - ylim[0]) * 0.2]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def zoom_out(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        new_xlim = [xlim[0] - (xlim[1] - xlim[0]) * 0.2, 
                   xlim[1] + (xlim[1] - xlim[0]) * 0.2]
        new_ylim = [ylim[0] - (ylim[1] - ylim[0]) * 0.2, 
                   ylim[1] + (ylim[1] - ylim[0]) * 0.2]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()
        
    def init_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StormLSTM(input_size=2, hidden_size=50, num_layers=2, output_size=2).to(device)
        self.model.load_pretrained()
        
    def predict_traj(self):
        # Xóa các điểm dự đoán trước đó
        for pred_point in self.predicted_points:
            pred_point.remove()
        self.predicted_points = []
        
        # Dự đoán quỹ đạo mới
        clicked_point = [[float(point[0]), float(point[1])] for point in self.clicked_points]
        predicted_points = self.model.predict_traj(clicked_point)
        for lon, lat in predicted_points:
            point, = self.ax.plot(lon, lat, 'go', transform=ccrs.PlateCarree())
            self.predicted_points.append(point)


if __name__ == "__main__":
    root = tk.Tk()
    app = StormMapApp(root)
    root.mainloop()