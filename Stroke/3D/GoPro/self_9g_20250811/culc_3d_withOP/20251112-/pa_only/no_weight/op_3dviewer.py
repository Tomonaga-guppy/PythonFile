import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
# OpenPose Body 25 Connectivity
SKELETON_PAIRS = [
    [0, 1],   # Nose - Neck
    [1, 8],   # Neck - MidHip
    [1, 2], [2, 3], [3, 4],       # Right Arm
    [1, 5], [5, 6], [6, 7],       # Left Arm
    [8, 9], [9, 10], [10, 11],    # Right Leg
    [8, 12], [12, 13], [13, 14],  # Left Leg
    [11, 24], [11, 22], [22, 23], # Right Foot
    [14, 21], [14, 19], [19, 20], # Left Foot
    [0, 15], [15, 17],            # Right Eye/Ear
    [0, 16], [16, 18]             # Left Eye/Ear
]

class MotionViewer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.frames = None
        self.is_playing = True
        self.current_frame = 0
        self.speed = 60  # Default Speed
        
        # Load Data
        self.load_data()
        
        # Setup Plot
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('3D Motion Viewer (Python)')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(bottom=0.2) # Make room for controls

        # Initial Plot Objects
        self.scatter_plot = self.ax.scatter([], [], [], c='cyan', s=20)
        self.lines_collection = Line3DCollection([], colors='white', linewidths=1.5)
        self.ax.add_collection(self.lines_collection)

        # Axis setup
        self.setup_axes()

        # Animation
        self.anim = FuncAnimation(self.fig, self.update, frames=None, 
                                  interval=1000/60, cache_frame_data=False)

        # --- Controls ---
        # Play/Stop Button
        ax_play = plt.axes([0.1, 0.05, 0.1, 0.05])
        self.btn_play = Button(ax_play, 'Stop')
        self.btn_play.on_clicked(self.toggle_play)

        # Frame Slider
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, len(self.frames)-1, valinit=0, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        # Speed Slider
        ax_speed = plt.axes([0.25, 0.05, 0.3, 0.03])
        self.slider_speed = Slider(ax_speed, 'Speed', 1, 120, valinit=60, valfmt='%d fps')
        self.slider_speed.on_changed(self.on_speed_change)

        plt.show()

    def load_data(self):
        """Load .npz file and select the best available data array."""
        try:
            print(f"Loading {self.filepath}...")
            with np.load(self.filepath) as data:
                if 'spline' in data:
                    print("Using 'spline' data.")
                    raw_data = data['spline']
                elif 'butter_filt' in data:
                    print("Using 'butter_filt' data.")
                    raw_data = data['butter_filt']
                elif 'raw' in data:
                    print("Using 'raw' data.")
                    raw_data = data['raw']
                else:
                    raise ValueError("No compatible keys (spline, butter_filt, raw) found in npz.")
                
                if raw_data.shape[1] != 25 or raw_data.shape[2] != 3:
                    print(f"Warning: Unexpected shape {raw_data.shape}. Visualize might be incorrect.")
                
                self.frames = raw_data
                self.calculate_bounds()

        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def calculate_bounds(self):
        """Calculate data bounds for auto-scaling and centering."""
        valid_points = self.frames[np.any(self.frames != 0, axis=2)]
        
        if len(valid_points) == 0:
            self.scale = 1
            self.center = np.array([0, 0, 0])
            return

        min_vals = np.min(valid_points, axis=0)
        max_vals = np.max(valid_points, axis=0)
        
        # Max range for scaling
        ranges = max_vals - min_vals
        max_range = np.max(ranges)
        
        self.scale = 1.0
        if max_range > 100: # likely mm
            self.scale = 0.001 # convert to meters
        
        print(f"Data Bounds: {min_vals} to {max_vals}")
        print(f"Auto-Scale Factor: {self.scale}")

    def transform_frame(self, frame_data):
        """
        Apply scaling and mapping.
        Matplotlib 3D rotates around the Z-axis.
        OpenPose data usually has Y as the vertical body axis.
        To allow natural 'turntable' rotation, we map Data-Y to Plot-Z.
        Values are kept native (no sign flipping).
        """
        points = frame_data * self.scale
        
        # Mapping: X->X, Y->Z, Z->Y
        # This makes the body stand upright in Matplotlib's coordinate system
        # allowing proper rotation around the body's vertical axis.
        new_points = np.zeros_like(points)
        new_points[:, 0] = points[:, 0] # X
        new_points[:, 1] = points[:, 2] # Z (mapped to Plot Y)
        new_points[:, 2] = points[:, 1] # Y (mapped to Plot Z - Vertical)
            
        return new_points

    def setup_axes(self):
        """Setup static 3D axes properties."""
        # Labels reflect the DATA axes, not the plot axes
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z (Depth)')
        self.ax.set_zlabel('Y (Vertical)')
        
        # Draw Origin Axis (Length 1m approx)
        # X (Red), Z (Blue->Y dir), Y (Green->Z dir)
        self.ax.plot([0, 1], [0, 0], [0, 0], c='r', label='X')
        self.ax.plot([0, 0], [0, 1], [0, 0], c='b', label='Z')
        self.ax.plot([0, 0], [0, 0], [0, 1], c='g', label='Y')
        
        # Z=0の平面を追加（薄い色で表示）
        limit = 2.0
        xx, yy = np.meshgrid(np.linspace(-limit, limit, 10), 
                             np.linspace(-limit, limit, 10))
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray', 
                            edgecolor='none', zorder=0)
        
        # Set consistent limits
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(0, 4)
        
        # Initial view
        self.ax.view_init(elev=10, azim=45)

    def update(self, frame):
        """Animation loop."""
        if self.is_playing:
            fps = 60
            step = max(1, int(self.slider_speed.val / fps * 2))
            self.current_frame = (self.current_frame + step) % len(self.frames)
            self.slider.set_val(self.current_frame)
        
        return self.update_plot(self.current_frame)

    def update_plot(self, frame_idx):
        """Redraw the skeleton for a specific frame."""
        raw_points = self.frames[frame_idx]
        points = self.transform_frame(raw_points)
        
        valid_mask = np.any(raw_points != 0, axis=1)
        
        # Update Joints
        self.scatter_plot._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        
        # Update Bones
        segments = []
        for start_idx, end_idx in SKELETON_PAIRS:
            if valid_mask[start_idx] and valid_mask[end_idx]:
                p1 = points[start_idx]
                p2 = points[end_idx]
                segments.append([p1, p2])
        
        self.lines_collection.set_segments(segments)
        return self.scatter_plot, self.lines_collection

    def toggle_play(self, event):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Stop' if self.is_playing else 'Play')

    def on_slider_change(self, val):
        self.current_frame = int(val)
        self.update_plot(self.current_frame)
        self.fig.canvas.draw_idle()

    def on_speed_change(self, val):
        pass

if __name__ == "__main__":
    target_file = None
    
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        print("Please select a .npz file...")
        root = tk.Tk()
        root.withdraw()
        target_file = filedialog.askopenfilename(
            title="Select Motion Data File",
            filetypes=[("NPZ Files", "*.npz"), ("All Files", "*.*")]
        )
        root.destroy()

    if target_file:
        print(f"Selected file: {target_file}")
        try:
            app = MotionViewer(target_file)
        except FileNotFoundError:
            print(f"Error: File '{target_file}' not found.")
    else:
        print("No file selected. Exiting.")