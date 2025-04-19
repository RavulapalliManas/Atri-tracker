import cv2
import numpy as np
import os
import argparse
import json
from tkinter import Tk, filedialog, messagebox

class BoxDrawer:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.image = None
        self.display_image = None
        self.original_image = None
        self.boxes = []  # List of boxes: [[(x1,y1), (x2,y2), (x3,y3), (x4,y4)], ...]
        self.box_colors = []  # Colors for each box
        self.active_box = -1
        self.active_corner = -1
        self.dragging = False
        self.drawing = False
        self.start_point = None
        self.window_name = "Box Drawer Tool"
        self.scale_factor = 1.0
        self.box_labels = []
        
        # Configuration
        self.corner_radius = 5
        self.line_thickness = 2
        self.text_scale = 0.6
        self.text_thickness = 2
        
    def load_image(self, image_path):
        """Load an image from a file path"""
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return False
            
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            print(f"Error: Could not read image at {image_path}")
            return False
            
        # Make a copy for display
        self.original_image = self.image.copy()
        self.display_image = self.image.copy()
        
        # Auto-scale large images for better display
        h, w = self.image.shape[:2]
        max_dimension = 1200  # Maximum width or height for display
        
        if max(h, w) > max_dimension:
            self.scale_factor = max_dimension / max(h, w)
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            self.display_image = cv2.resize(self.image, (new_w, new_h))
            print(f"Image scaled by factor {self.scale_factor:.2f} for display")
        else:
            self.scale_factor = 1.0
            
        return True
        
    def load_video_frame(self, video_path, frame_num=0):
        """Load a frame from a video file"""
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return False
            
        # Seek to the desired frame
        if frame_num > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: Could not read frame {frame_num} from video")
            return False
            
        self.image_path = video_path
        self.image = frame
        self.original_image = frame.copy()
        self.display_image = frame.copy()
        
        # Auto-scale large frames
        h, w = self.image.shape[:2]
        max_dimension = 1200
        
        if max(h, w) > max_dimension:
            self.scale_factor = max_dimension / max(h, w)
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            self.display_image = cv2.resize(self.image, (new_w, new_h))
            print(f"Frame scaled by factor {self.scale_factor:.2f} for display")
        else:
            self.scale_factor = 1.0
            
        return True
        
    def scale_point(self, point, reverse=False):
        """Scale a point between display and original image coordinates"""
        if reverse:  # Display to original
            return (int(point[0] / self.scale_factor), int(point[1] / self.scale_factor))
        else:  # Original to display
            return (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
            
    def add_box(self, box, label=""):
        """Add a new box"""
        self.boxes.append(box)
        self.box_labels.append(label or f"Box {len(self.boxes)}")
        
        # Assign a random color to the box
        color = (
            np.random.randint(50, 200),
            np.random.randint(50, 200),
            np.random.randint(50, 200)
        )
        self.box_colors.append(color)
        
    def update_display(self):
        """Update the display image with all boxes and handles"""
        if self.image is None:
            return
            
        # Start with a clean copy
        self.display_image = self.original_image.copy()
        if self.scale_factor != 1.0:
            h, w = self.original_image.shape[:2]
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            self.display_image = cv2.resize(self.original_image, (new_w, new_h))
            
        # Draw all boxes
        for i, box in enumerate(self.boxes):
            # Scale the box coordinates for display
            display_box = [self.scale_point(p) for p in box]
            
            # Draw the polygon
            color = self.box_colors[i]
            cv2.polylines(
                self.display_image,
                [np.array(display_box, dtype=np.int32)],
                True,
                color,
                self.line_thickness
            )
            
            # Draw the label
            label_pos = (
                int(sum(p[0] for p in display_box) / 4),  # Average x
                int(sum(p[1] for p in display_box) / 4)   # Average y
            )
            cv2.putText(
                self.display_image,
                self.box_labels[i],
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                color,
                self.text_thickness
            )
            
            # Draw corners/handles
            for j, point in enumerate(display_box):
                cv2.circle(
                    self.display_image,
                    point,
                    self.corner_radius,
                    color,
                    -1 if i == self.active_box and j == self.active_corner else 2
                )
                
            # Coordinates display
            for j, point in enumerate(box):  # Use original coordinates
                coord_text = f"({point[0]}, {point[1]})"
                text_pos = (display_box[j][0] + 10, display_box[j][1] + 10)
                cv2.putText(
                    self.display_image,
                    coord_text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
                
        # If currently drawing a new box
        if self.drawing and self.start_point:
            cv2.circle(
                self.display_image,
                self.scale_point(self.start_point),
                self.corner_radius,
                (0, 255, 0),
                -1
            )
            
    def find_nearest_corner(self, point):
        """Find the nearest box corner to the given point"""
        min_distance = float('inf')
        nearest_box = -1
        nearest_corner = -1
        
        for i, box in enumerate(self.boxes):
            for j, corner in enumerate(box):
                display_corner = self.scale_point(corner)
                distance = np.sqrt(
                    (point[0] - display_corner[0])**2 + 
                    (point[1] - display_corner[1])**2
                )
                
                if distance < min_distance and distance < 15:  # 15 pixel threshold
                    min_distance = distance
                    nearest_box = i
                    nearest_corner = j
                    
        return nearest_box, nearest_corner
                
    def is_point_in_box(self, point, box_idx):
        """Check if a point is inside a box"""
        if box_idx < 0 or box_idx >= len(self.boxes):
            return False
            
        box = self.boxes[box_idx]
        display_box = [self.scale_point(p) for p in box]
        polygon = np.array(display_box, dtype=np.int32)
        
        return cv2.pointPolygonTest(polygon, point, False) >= 0
                
    def handle_mouse_event(self, event, x, y, flags, param):
        """Handle mouse events for drawing and adjusting boxes"""
        if self.image is None:
            return
            
        current_point = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on an existing corner
            box_idx, corner_idx = self.find_nearest_corner(current_point)
            
            if box_idx >= 0:
                # Start dragging a corner
                self.active_box = box_idx
                self.active_corner = corner_idx
                self.dragging = True
            else:
                # Check if clicking inside a box
                for i in range(len(self.boxes)):
                    if self.is_point_in_box(current_point, i):
                        self.active_box = i
                        self.active_corner = -1  # -1 means entire box
                        self.dragging = True
                        self.drag_start = current_point
                        self.box_start = self.boxes[i].copy()
                        break
                else:
                    # Start drawing a new box
                    self.drawing = True
                    self.start_point = self.scale_point(current_point, True)  # Convert to original coords
                    
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.active_box >= 0:
                if self.active_corner >= 0:
                    # Dragging a corner
                    self.boxes[self.active_box][self.active_corner] = self.scale_point(current_point, True)
                else:
                    # Dragging the entire box
                    dx = current_point[0] - self.drag_start[0]
                    dy = current_point[1] - self.drag_start[1]
                    
                    for i in range(len(self.boxes[self.active_box])):
                        orig_x, orig_y = self.box_start[i]
                        orig_display_x, orig_display_y = self.scale_point((orig_x, orig_y))
                        new_display_x = orig_display_x + dx
                        new_display_y = orig_display_y + dy
                        self.boxes[self.active_box][i] = self.scale_point((new_display_x, new_display_y), True)
                        
            self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                # Finish drawing a new box
                end_point = self.scale_point(current_point, True)  # Convert to original coords
                
                # Create a box with 4 corners
                x1, y1 = self.start_point
                x2, y2 = end_point
                
                new_box = [
                    (x1, y1),  # Top-left
                    (x2, y1),  # Top-right
                    (x2, y2),  # Bottom-right
                    (x1, y2)   # Bottom-left
                ]
                
                self.add_box(new_box)
                self.start_point = None
                self.drawing = False
                
            self.dragging = False
            self.active_corner = -1
            
            self.update_display()
            
    def run(self):
        """Run the interactive box drawing tool"""
        if self.image is None:
            print("No image loaded. Please load an image first.")
            return
            
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.handle_mouse_event)
        
        print("\nBox Drawing Tool Controls:")
        print("- Left-click and drag to draw a new box")
        print("- Click near a corner and drag to move it")
        print("- Click inside a box and drag to move the entire box")
        print("- Press 'r' to reset/remove all boxes")
        print("- Press 'd' to delete the most recently selected box")
        print("- Press 'n' to add a label to the selected box")
        print("- Press 's' to save the box coordinates to a file")
        print("- Press 'l' to load box coordinates from a file")
        print("- Press 'q' to quit\n")
        
        self.update_display()
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.boxes = []
                self.box_colors = []
                self.box_labels = []
                self.update_display()
            elif key == ord('s'):
                self.save_boxes()
            elif key == ord('l'):
                self.load_boxes()
            elif key == ord('n'):
                if self.active_box >= 0 and self.active_box < len(self.boxes):
                    label = input(f"Enter label for box {self.active_box+1}: ")
                    self.box_labels[self.active_box] = label
                    self.update_display()
            elif key == ord('d'):
                if self.active_box >= 0 and self.active_box < len(self.boxes):
                    self.boxes.pop(self.active_box)
                    self.box_colors.pop(self.active_box)
                    self.box_labels.pop(self.active_box)
                    self.active_box = -1
                    self.update_display()
                    
        cv2.destroyAllWindows()
        
    def save_boxes(self):
        """Save box coordinates to a JSON file"""
        if not self.boxes:
            print("No boxes to save")
            return
            
        root = Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        root.destroy()
        
        if not file_path:
            return
            
        # Prepare data for saving
        box_data = {}
        
        # Add tank coordinates
        tank_coords = []
        for box in self.boxes:
            if self.box_labels[self.boxes.index(box)].lower() == "tank":
                tank_coords = box
                break
        
        if tank_coords:
            box_data["tank_coordinates"] = tank_coords
        
        # Add all boxes
        for i, box in enumerate(self.boxes):
            label = self.box_labels[i]
            if label.lower() != "tank":  # Skip tank box as it's handled separately
                box_data[label] = {
                    "coords": box
                }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(box_data, f, indent=2)
            print(f"Saved box coordinates to {file_path}")
            
            # Show success message
            root = Tk()
            root.withdraw()
            messagebox.showinfo("Save Successful", f"Saved box coordinates to {file_path}")
            root.destroy()
        except Exception as e:
            print(f"Error saving box coordinates: {e}")
            
    def load_boxes(self):
        """Load box coordinates from a JSON file"""
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        root.destroy()
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                box_data = json.load(f)
                
            # Reset current boxes
            self.boxes = []
            self.box_colors = []
            self.box_labels = []
            
            # Load tank coordinates if available
            if "tank_coordinates" in box_data:
                self.add_box(box_data["tank_coordinates"], "Tank")
            
            # Load all other boxes
            for label, data in box_data.items():
                if label != "tank_coordinates":
                    if isinstance(data, dict) and "coords" in data:
                        self.add_box(data["coords"], label)
                    else:
                        # Handle legacy format
                        self.add_box(data, label)
                        
            print(f"Loaded {len(self.boxes)} boxes from {file_path}")
            self.update_display()
        except Exception as e:
            print(f"Error loading box coordinates: {e}")
            
    def get_box_data(self):
        """Return the box data in a format compatible with the tracking system"""
        box_data = {}
        
        # Add all boxes
        for i, box in enumerate(self.boxes):
            label = self.box_labels[i]
            if label.lower() != "tank":  # Skip tank box as it's handled separately
                box_data[label] = {
                    "coords": box
                }
        
        return box_data
        
    def get_tank_coordinates(self):
        """Return the tank coordinates if defined"""
        for i, label in enumerate(self.box_labels):
            if label.lower() == "tank":
                return self.boxes[i]
        return None

def main():
    parser = argparse.ArgumentParser(description='Interactive Box Drawing Tool')
    parser.add_argument('--image', help='Path to image file (optional)')
    parser.add_argument('--video', help='Path to video file (optional)')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to load from video (default: 0)')
    
    args = parser.parse_args()
    
    drawer = BoxDrawer()
    
    if args.image:
        drawer.load_image(args.image)
    elif args.video:
        drawer.load_video_frame(args.video, args.frame)
    else:
        # No image/video specified, let user select
        root = Tk()
        root.withdraw()
        
        choice = input("Load from (1) Image or (2) Video? [1/2]: ")
        
        if choice == '1':
            file_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                drawer.load_image(file_path)
        elif choice == '2':
            file_path = filedialog.askopenfilename(
                title="Select a video",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                frame_num = input("Enter frame number to load (default: 0): ")
                try:
                    frame_num = int(frame_num)
                except ValueError:
                    frame_num = 0
                drawer.load_video_frame(file_path, frame_num)
        
        root.destroy()
    
    if drawer.image is not None:
        drawer.run()
    else:
        print("No image or video loaded. Exiting.")

if __name__ == "__main__":
    main() 