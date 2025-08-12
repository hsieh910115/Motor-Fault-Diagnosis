This project performs motor fault diagnosis using the University of Ottawa Electric Motor Dataset. 

The dataset contains one healthy class and seven fault classes. Each class includes eight operating conditions, further divided into loaded and no-load, and each file is 10 seconds long.

I segment each recording into 100 samples and convert them into time–frequency spectrograms via STFT.

The spectrograms are sent from Python to the MCU over UART, upon receiving the data, the MCU performs diagnosis using an on-device CNN together with the inference logic implemented in main.c.

In the master branch is the complete STM32 project, and in the python branch is the Python code.
