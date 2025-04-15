import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from model import StormLSTM
import torch
import random
from shapely.geometry import Point
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


class StormMapApp:
    def __init__(self, root, trajectory_test):
        self.root = root
        self.root.title("Storm Trajectory Map")
        
        # Lưu trajectory_test vào self
        self.trajectory_test = trajectory_test
        self.init_model()
        
        # Danh sách các điểm được click, các đường nối và các điểm dự đoán
        self.clicked_points = []
        self.red_lines = []
        self.predicted_points = []
        self.storm_start_points = []
        self.random_trajectory_lines = []
        
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
        
        # Biến trạng thái cho nút switch
        self.show_storm_starts = tk.BooleanVar(value=False)
        self.show_random_traj = tk.BooleanVar(value=False)
        
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
        
        # Thêm nút switch
        tk.Checkbutton(button_frame, text="Show Storm Starts", variable=self.show_storm_starts,
                      command=self.toggle_storm_starts).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(button_frame, text="Show Random Trajectory", variable=self.show_random_traj,
                      command=self.toggle_random_trajectory).pack(side=tk.LEFT, padx=5)

    def update_cursor(self):
        # Thay đổi con trỏ chuột dựa trên mode
        cursor = "crosshair" if self.mode.get() == "click" else "hand2"
        self.canvas.get_tk_widget().config(cursor=cursor)

    def is_point_over_ocean(self, lon, lat):
        # Kiểm tra xem điểm có nằm trên đại dương hay không
        point = Point(lon, lat)
        land = cfeature.LAND.geometries()
        for geom in land:
            if geom.contains(point):
                return False
        return True

    def toggle_storm_starts(self):
        # Xóa các điểm bắt đầu hiện tại
        for point in self.storm_start_points:
            point.remove()
        self.storm_start_points = []
        
        if self.show_storm_starts.get():
            # Hiển thị các điểm bắt đầu của các trajectory trong trajectory_test
            for traj in self.trajectory_test:
                if traj:  # Kiểm tra trajectory không rỗng
                    first_point = traj[0]  # Lấy điểm đầu tiên
                    lon, lat = float(first_point[0]), float(first_point[1])
                    # Chỉ hiển thị nếu điểm nằm trên đại dương và trong phạm vi bản đồ
                    if (100 <= lon <= 180 and -10 <= lat <= 50 and 
                        self.is_point_over_ocean(lon, lat)):
                        point, = self.ax.plot(lon, lat, 'bo', transform=ccrs.PlateCarree())
                        self.storm_start_points.append(point)
        
        self.canvas.draw()

    def toggle_random_trajectory(self):
        # Xóa quỹ đạo hiện tại
        for line in self.random_trajectory_lines:
            line.remove()
        self.random_trajectory_lines = []
        
        if self.show_random_traj.get():
            # Lấy mẫu ngẫu nhiên một quỹ đạo từ trajectory_test
            if self.trajectory_test:
                random_traj = random.choice(self.trajectory_test)
                if random_traj:
                    lons = [float(point[0]) for point in random_traj]
                    lats = [float(point[1]) for point in random_traj]
                    # Chỉ vẽ nếu quỹ đạo có điểm trong phạm vi bản đồ
                    if any((100 <= lon <= 180 and -10 <= lat <= 50) 
                           for lon, lat in zip(lons, lats)):
                        line, = self.ax.plot(lons, lats, color='blue', linewidth=1,
                                           transform=ccrs.PlateCarree())
                        self.random_trajectory_lines.append(line)
        
        self.canvas.draw()

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
        self.model = StormLSTM().to(device)
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


# Example usage:
# Assuming trajectory_test is a list of trajectories, where each trajectory is a list of [lon, lat] points
if __name__ == "__main__":
    root = tk.Tk()
    # Example trajectory_test (replace with your actual data)
    trajectories, offsets, long_scaler, lati_scaler, delta_lon_scaler, delta_lat_scaler = get_data()
    trajectories_train, trajectories_test, offsets_train, offsets_test = train_test_split(
        trajectories, offsets, test_size=0.2, random_state=42
    )
    app = StormMapApp(root, trajectories_test)
    root.mainloop()