
# **Iris-to-Pupil Ratio Detection using Daugman Segmentation**  

## **📌 Introduction**  
This project implements **Daugman's Segmentation** algorithm for **iris and pupil segmentation**, complemented by **OpenCV's Haar cascades** for initial eye detection. The system detects the **iris-to-pupil ratio** in real-time using a webcam feed.  

## **📝 Features**  
✅ Real-time **face and eye detection** using OpenCV's Haar cascades  
✅ **Daugman’s algorithm** for accurate **iris and pupil segmentation**  
✅ Computation of **iris-to-pupil ratio**  
✅ Live visualization of segmentation results on the video feed  

## **📂 Project Structure**  
```
📦 Iris_Pupil_Detection
 ┣ 📂 irisSeg/              # Pre-trained models for Daugman-segmentation
 ┣ 📂 data/                 # Sample images and dataset
 ┣ 📂 scripts/              # Core Python scripts for real-time detection
 ┣ 📜 requirements.txt      # Dependencies for the project
 ┣ 📜 README.md             # Project documentation
 ┗ 📜 main.py               # Main executable script


## **⚙️ Installation**  
1️⃣ Clone the repository  
```bash
git clone https://github.com/Shyam0624/iris-pupil-ratio-using-daugman-segmentation
cd Iris-Pupil-Detection
```
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
3️⃣ Run the program  
```bash
python main.py
```

## **🖥️ Implementation Steps**  
1. **Face & Eye Detection** 🧐  
   - Uses **Haar cascades** for face and eye recognition.  

2. **Iris & Pupil Segmentation** 👁️  
   - **Daugman’s operator** detects circular boundaries for **iris and pupil**.  

3. **Iris-to-Pupil Ratio Calculation** 📊  
   - Computes ratio as:  
     \[
     \text{Ratio} = \frac{\text{Iris Radius}}{\text{Pupil Radius}}
     \]  

4. **Real-time Visualization** 🎥  
   - Annotates the **segmentation results and ratio** on the video feed.  


## **📸 Sample Results**  
| Image | Iris-to-Pupil Ratio |
|-------|---------------------|
| 1.![image](https://github.com/user-attachments/assets/1b3ce288-396c-4218-a6a2-2a6a64ab1b3a) | **3.11** |
| 2.![image](https://github.com/user-attachments/assets/02000de8-4c31-4731-9382-8edc96f14ddc) | **3.06** |
| 3.![image](https://github.com/user-attachments/assets/e0eb304a-4a61-4c5c-b973-9698b07951c8) | **2.41** |
