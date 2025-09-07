# Motor Fault Diagnosis on STM32 with AI

This project implements **motor fault diagnosis** using the **University of Ottawa Electric Motor Dataset**.  
The workflow integrates **Python for preprocessing** and **MCU for on-device inference**.

MCU: STM32F407

## 🔹 Dataset
- **Classes**: 1 healthy + 7 fault classes  
- **Operating conditions**: 8 per class  
- **Load conditions**: loaded / no-load  
- **File length**: 10 seconds each  



## 🔹 Data Processing
1. Each 10-second recording is segmented into **100 samples**.  
2. Signals are converted into **time–frequency spectrograms** using **STFT**.  
3. Spectrograms are sent from **Python → STM32 MCU** via **UART**.  



## 🔹 On-Device Diagnosis
- The MCU runs a **CNN-based classifier**.  
- Inference logic is implemented in **`main.c`**.  
- Supports **real-time motor fault detection** on embedded hardware.  



## 🔹 Project Structure
- **`master` branch** → Complete STM32 project  
- **`python` branch** → Python preprocessing and communication scripts  



## 🔹 Demo Screenshots

### ✅ Waiting for STFT images
<img width="598" height="301" alt="132132" src="https://github.com/user-attachments/assets/2004a691-e4a9-41fe-bc73-a0c136c75271" />

### 🔌 MCU Diagnosis Result
<img width="236" height="213" alt="board1" src="https://github.com/user-attachments/assets/d7e5b568-44ef-48a1-b2d7-8de79d2e7154" />
<img width="220" height="214" alt="board2" src="https://github.com/user-attachments/assets/c677160e-ada9-45ed-a751-060515095fe4" />



