import sys
import time
import cv2
from dummydaugman import irisSeg  # Ensure irisSeg.py is in the correct directory
import numpy as np
import os


class EyeTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.last_capture_time = 0
        self.capture_interval = 1  # seconds between captures
        self.last_saved_time = 0  # Time when the last frame was saved
        self.save_interval = 1  # Save frame every 1 seconds
        self.left_ratio = None
        self.right_ratio = None

        # Ensure the capture_frame directory exists
        if not os.path.exists("capture_frame"):
            os.makedirs("capture_frame")

    def process_eye_region(self, eye_roi):
        # Convert to grayscale if not already
        if len(eye_roi.shape) > 2:
            eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        eye_roi = cv2.equalizeHist(eye_roi)
        
        try:
            # Perform iris segmentation
            coord_iris, coord_pupil, _ = irisSeg(eye_roi, rmin=20, rmax=35, view_output=False)
            
            # Calculate ratio
            if coord_iris[2] != 0:  # Prevent division by zero
                ratio = coord_iris[2] / coord_pupil[2]
                return ratio, coord_iris, coord_pupil
        except Exception as e:
            print(f"Error in eye processing: {e}")
        
        return None, None, None

    def draw_measurements(self, frame, eye_box, coords_iris, coords_pupil):
        x, y, w, h = eye_box
        if coords_iris is not None and coords_pupil is not None:

            iris_radius = int(coords_iris[2].item()*0.3) # Reduce the iris radius by 20%
            pupil_radius = int(coords_pupil[2].item()*0.3)  # Reduce the pupil radius by 20%
            # Draw iris circle
            cv2.circle(frame,
                    (int(x + coords_iris[1].item()), int(y + coords_iris[0].item())),
                    iris_radius,
                    (0, 255, 0),
            2)

            # Draw pupil circle
            cv2.circle(frame,
                      (int(x + coords_pupil[1].item()), int(y + coords_pupil[0].item())),
                       pupil_radius,
                      (255, 0, 0),
                    2)

    
    def process_frame(self, cropped_frame):
        """ # Define a smaller region (for example, center of the frame)
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        region_width, region_height = w // 2, h // 2  # Define smaller region (50% of original frame size)
        
        # Crop the frame to focus on a smaller region
        cropped_frame = frame[center_y - region_height // 2: center_y + region_height // 2,
                              center_x - region_width // 2: center_x + region_width // 2]"""
        
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        current_time = time.time()
        should_capture = (current_time - self.last_capture_time) >= self.capture_interval
        should_save = (current_time - self.last_saved_time) >= self.save_interval
        

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = cropped_frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            # Sort eyes by x-coordinate to distinguish left and right
            eyes = sorted(eyes, key=lambda e: e[0])

            # Initialize the ratios as None (indicating no detection)
            self.left_ratio = None
            self.right_ratio = None
            self.left_iris_radius = 0  # Initialize iris radius for left eye
            self.right_iris_radius = 0  # Initialize iris radius for right eye
            self.left_pupil_radius = 0  # Initialize pupil radius for left eye
            self.right_pupil_radius = 0  # Initialize pupil radius for right eye

            if len(eyes) >= 1:  # At least one eye detected
                for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Iterate over up to 2 eyes
                    eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]

                    if should_capture:
                        ratio, coords_iris, coords_pupil = self.process_eye_region(eye_roi)

                        # Update the ratio for the left or right eye
                        if ratio is not None:
                            if i == 0:  # Left eye
                                self.left_ratio = ratio
                                self.left_iris_radius = int(coords_iris[2].item())  # Extract iris radius in px
                                self.left_pupil_radius = int(coords_pupil[2].item())  # Extract pupil radius in px
                            else:  # Right eye
                                self.right_ratio = ratio
                                self.right_iris_radius = int(coords_iris[2].item())  # Extract iris radius in px
                                self.right_pupil_radius = int(coords_pupil[2].item())  # Extract pupil radius in px

                        # Draw measurements (only iris and pupil circles)
                        self.draw_measurements(roi_color, (ex, ey, ew, eh), coords_iris, coords_pupil)

            # Handle the case where no eye is detected
            if self.left_ratio is None:
                self.left_ratio = 0.00  # Set left ratio to zero if not detected

            if self.right_ratio is None:
                self.right_ratio = 0.00  # Set right ratio to zero if not detected

            # Optionally display the iris/pupil ratios on the frame
            # Display Left Eye Ratio
            left_ratio = self.left_ratio.item() if isinstance(self.left_ratio, np.ndarray) else self.left_ratio
            cv2.putText(cropped_frame, f"Left Eye I/P Ratio: {left_ratio:.3f}",  #Blue
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Display Right Eye Ratio
            right_ratio = self.right_ratio.item() if isinstance(self.right_ratio, np.ndarray) else self.right_ratio
            cv2.putText(cropped_frame, f"Right Eye I/P Ratio: {right_ratio:.3f}", #Red
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
             # Display iris and pupil radii (in pixels) at the bottom of the frame
            cv2.putText(cropped_frame, f"Left Iris Radius: {self.left_iris_radius} px",
                        (10, cropped_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(cropped_frame, f"Left Pupil Radius: {self.left_pupil_radius} px",
                        (10, cropped_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.putText(cropped_frame, f"Right Iris Radius: {self.right_iris_radius} px",
                        (250, cropped_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(cropped_frame, f"Right Pupil Radius: {self.right_pupil_radius} px",
                        (250, cropped_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            if should_save:
                timestamp = int(time.time())
                save_path = f"capture_frame/frame_{timestamp}.jpg"
                cv2.imwrite(save_path, cropped_frame)
                print(f"Saved frame to {save_path}")
                self.last_saved_time = current_time

        if should_capture:
            self.last_capture_time = current_time

        return cropped_frame
    

def main():
    cap = cv2.VideoCapture(0)
    tracker = EyeTracker()
    
    while True:
        ret, cropped_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        processed_frame = tracker.process_frame(cropped_frame)
        
        # Display result
        cv2.imshow('Eye Tracking', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
