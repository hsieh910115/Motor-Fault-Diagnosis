This project performs motor fault diagnosis using the University of Ottawa Electric Motor Dataset. 

The dataset contains one healthy class and seven fault classes. Each class includes eight operating conditions, further divided into loaded and no-load, and each file is 10 seconds long.

I segment each recording into 100 samples and convert them into time–frequency spectrograms via STFT.

The spectrograms are sent from Python to the MCU over UART, upon receiving the data, the MCU performs diagnosis using an on-device CNN together with the inference logic implemented in main.c.

In the master branch is the complete STM32 project, and in the python branch is the Python code.



 <img width="400" height="236" alt="image" src="https://github.com/user-attachments/assets/3c7ac6c6-b27e-473c-a099-6d1977b6c647" />  <img width="236" height="213" alt="image" src="https://github.com/user-attachments/assets/d7e5b568-44ef-48a1-b2d7-8de79d2e7154" />  <img width="220" height="214" alt="image" src="https://github.com/user-attachments/assets/c677160e-ada9-45ed-a751-060515095fe4" />



