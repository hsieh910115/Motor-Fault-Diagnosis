first_stage.ipynb: This script processes a public dataset using STFT, then feeds the results into a CNN model for training and testing. Finally, the model is converted into the TFLite format so that it can be deployed to STM32 using X-Cube-AI.

ttl.py: This script sends time–frequency spectrograms to the MCU via UART.
