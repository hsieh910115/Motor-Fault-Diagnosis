This project performs motor fault diagnosis using the University of Ottawa Electric Motor Dataset. 

The dataset contains one healthy class and seven fault classes. Each class includes eight operating conditions, further divided into loaded and no-load, and each file is 10 seconds long.

I segment each recording into 100 samples and convert them into time–frequency spectrograms via STFT.

The spectrograms are sent from Python to the MCU over UART, upon receiving the data, the MCU performs diagnosis using an on-device CNN together with the inference logic implemented in main.c.

In the master branch is the complete STM32 project, and in the python branch is the Python code.

<img width="362" height="182" alt="image" src="https://github.com/user-attachments/assets/54a1c9dd-9456-4a15-a935-8d538bfab22a" />

<img width="246" height="225" alt="image" src="https://github.com/user-attachments/assets/27c02a5e-4aab-46b7-ac06-1a3259e46b4d" /> <img width="230" height="226" alt="image" src="https://github.com/user-attachments/assets/26abbf40-1b43-4c78-ae4f-858dafe7543b" />

