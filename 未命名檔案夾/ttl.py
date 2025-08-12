import serial
import numpy as np
import pandas as pd
from scipy import signal
from skimage.transform import resize
import matplotlib.pyplot as plt

def load_fault_stft_sample(fault_type='H_H', condition='4_0', sample_index=0):
    # 設定參數（與STFT程式碼一致）
    fs = 42000
    window_size = 512
    overlap = 460
    window = signal.windows.hann(window_size)    
    
    # 檔案路徑
    file_name = f"/Users/zongyan/Desktop/EMTRC/sound/UOEMD_VAFCVS/2_CSV_Data_Files/{fault_type}_{condition}.csv"
    
    try:
        # 讀取數據
        data = pd.read_csv(file_name)
        acoustic_signal = data.iloc[:, 1].values
        
        # 計算樣本長度
        total_length = len(acoustic_signal)
        sample_length = total_length // 100
        
        # 取指定的樣本
        start_idx = sample_index * sample_length
        end_idx = (sample_index + 1) * sample_length
        sample = acoustic_signal[start_idx:end_idx]
        
        # 計算STFT
        f, t, Zxx = signal.stft(sample, fs=fs, window=window, 
                              nperseg=window_size, noverlap=overlap, 
                              nfft=window_size, return_onesided=True)
        
        spectrogram = np.abs(Zxx)
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        # 限制頻率範圍到2000Hz
        freq_limit = 2000
        freq_mask = f <= freq_limit
        limited_spectrogram = spectrogram_db[freq_mask, :]
        
        # 調整大小為64x64（與您的CNN輸入一致）
        resized_spectrogram = resize(limited_spectrogram, (64, 64), anti_aliasing=True)
        
        return resized_spectrogram, fault_type
        
    except FileNotFoundError:
        print(f"警告：找不到檔案 {file_name}")
        return None, None


def send_stft_to_mcu(stft_data, fault_type, port='/dev/tty.usbserial-10'):
    import time
    
    try:
        # 建立串列連接，增加緩衝區大小
        ser = serial.Serial(port, 115200, timeout=10)  # 增加 timeout
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        print(f"已連接到 {port}")
        
        # 在發送數據前先讀取可能的初始化訊息
        time.sleep(1)
        if ser.in_waiting > 0:
            initial_response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"初始化訊息: {initial_response}")
        
        # 數據預處理
        stft_flat = stft_data.flatten()
        stft_normalized = (stft_flat - np.min(stft_flat)) / (np.max(stft_flat) - np.min(stft_flat))
        stft_data_uint8 = (stft_normalized * 255).astype(np.uint8)
        
        print(f"準備發送 {len(stft_data_uint8)} 個數據點")
        
        # 發送開始標記
        ser.write(bytes([0xFF]))
        ser.flush()
        print("已發送開始標記")
        
        # 分批發送數據
        batch_size = 32
        for i in range(0, len(stft_data_uint8), batch_size):
            batch = stft_data_uint8[i:i+batch_size]
            ser.write(batch)
            ser.flush()
            time.sleep(0.001)
            
            if i % 1000 == 0:
                print(f"已發送 {i}/{len(stft_data_uint8)} 個數據點")
        
        # 發送結束標記
        ser.write(bytes([0xFE]))
        ser.flush()
        print("已發送結束標記")
        
        # 等待並接收診斷結果
        diagnosis_result = receive_diagnosis_result(ser, timeout=15)
        
        ser.close()
        return diagnosis_result
        
    except Exception as e:
        print(f"發生錯誤: {e}")
        return None

def receive_diagnosis_result(ser, timeout=15):
    """
    接收 MCU 的診斷結果
    """
    import time
    
    print("等待MCU診斷結果...")
    start_time = time.time()
    response_buffer = ""
    
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            try:
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                response_buffer += data
                print(f"接收到數據: {data.strip()}")
                
                # 檢查是否收到診斷結果
                lines = response_buffer.split('\n')
                for line in lines:
                    line = line.strip()
                    
                    # 檢查診斷結果格式
                    if "DIAGNOSIS:" in line:
                        if "FAULT" in line:
                            # 提取信心度
                            try:
                                confidence_start = line.find("confidence:") + 11
                                confidence_end = line.find(")", confidence_start)
                                confidence = float(line[confidence_start:confidence_end])
                                return {
                                    'status': 'FAULT',
                                    'confidence': confidence,
                                    'raw_response': line
                                }
                            except:
                                return {
                                    'status': 'FAULT',
                                    'confidence': 'unknown',
                                    'raw_response': line
                                }
                        elif "HEALTHY" in line:
                            try:
                                confidence_start = line.find("confidence:") + 11
                                confidence_end = line.find(")", confidence_start)
                                confidence = float(line[confidence_start:confidence_end])
                                return {
                                    'status': 'HEALTHY',
                                    'confidence': confidence,
                                    'raw_response': line
                                }
                            except:
                                return {
                                    'status': 'HEALTHY',
                                    'confidence': 'unknown',
                                    'raw_response': line
                                }
                    
                    # 檢查結果格式（如果使用 RESULT: 格式）
                    elif "RESULT:" in line:
                        try:
                            result_parts = line.replace("RESULT:", "").split(',')
                            diagnosis = result_parts[0].strip()
                            confidence = float(result_parts[1].strip())
                            
                            if "FAULT" in diagnosis:
                                return {
                                    'status': 'FAULT',
                                    'confidence': confidence,
                                    'raw_response': line
                                }
                            else:
                                return {
                                    'status': 'HEALTHY',
                                    'confidence': confidence,
                                    'raw_response': line
                                }
                        except:
                            return {
                                'status': 'UNKNOWN',
                                'confidence': 'unknown',
                                'raw_response': line
                            }
            except Exception as e:
                print(f"解碼錯誤: {e}")
        
        time.sleep(0.1)  # 短暫等待
    
    print("診斷結果接收超時")
    return {
        'status': 'TIMEOUT',
        'confidence': 'unknown',
        'raw_response': response_buffer
    }

def analyze_diagnosis_result(result, expected_fault_type):
    """
    分析診斷結果並與預期結果比較
    """
    if result is None:
        print("❌ 無法獲得診斷結果")
        return False
    
    print(f"\n📊 診斷結果分析:")
    print(f"   狀態: {result['status']}")
    print(f"   信心度: {result['confidence']}")
    print(f"   原始回應: {result['raw_response']}")
    
    # 判斷預期結果
    if expected_fault_type in ['S_W', 'R_U', 'R_M', 'V_U', 'B_R', 'K_A', 'F_B']:  # 故障類型
        expected_status = 'FAULT'
        expected_led = '🔴 紅燈'
    else:  # 健康類型 (H_H)
        expected_status = 'HEALTHY'
        expected_led = '🟢 綠燈'
    
    print(f"\n🎯 預期結果:")
    print(f"   故障類型: {expected_fault_type}")
    print(f"   預期狀態: {expected_status}")
    print(f"   預期LED: {expected_led}")
    
    # 比較結果
    if result['status'] == expected_status:
        print(f"✅ 診斷正確！MCU正確識別了{expected_fault_type}")
        return True
    elif result['status'] == 'TIMEOUT':
        print(f"⏰ 診斷超時，請檢查MCU狀態")
        return False
    else:
        print(f"❌ 診斷錯誤！預期{expected_status}，但得到{result['status']}")
        return False


def visualize_stft_before_sending(stft_data, fault_type):
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(stft_data, aspect='auto', origin='lower', cmap='jet')
    plt.title(f'Original STFT - {fault_type}')
    plt.colorbar(label='Magnitude (dB)')
    
    # 正規化後的數據
    stft_normalized = (stft_data - np.min(stft_data)) / (np.max(stft_data) - np.min(stft_data))
    plt.subplot(2, 2, 2)
    plt.imshow(stft_normalized, aspect='auto', origin='lower', cmap='jet')
    plt.title(f'Normalized STFT (0-1) - {fault_type}')
    plt.colorbar(label='Normalized Magnitude')
    
    # 轉換為uint8後的數據
    stft_uint8 = (stft_normalized * 255).astype(np.uint8)
    plt.subplot(2, 2, 3)
    plt.imshow(stft_uint8, aspect='auto', origin='lower', cmap='jet')
    plt.title(f'STFT as uint8 (0-255) - {fault_type}')
    plt.colorbar(label='uint8 Value')
    
    # 數據分布直方圖
    plt.subplot(2, 2, 4)
    plt.hist(stft_uint8.flatten(), bins=50, alpha=0.7)
    plt.title(f'Data Distribution - {fault_type}')
    plt.xlabel('uint8 Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'stft_visualization_{fault_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函數 - 測試不同故障類型並收集結果
    """
    # 定義要測試的故障類型
    fault_types_to_test = [
        ('H_H', 'Healthy '),      
        ('S_W', 'Stator Winding Fault'),    
        ('R_U', 'Rotor Unbalance'),    
        ('R_M', 'Rotor Misalignment'),      
        ('V_U', 'Voltage Unbalance'),
        ('B_R', 'Bowed Rotor'),
        ('K_A', 'Broken Rotor Bars'),
        ('R_B', 'Faulty Bearings'),               
    ]
    
    # 結果統計
    test_results = []
    correct_count = 0
    total_count = 0
    
    for fault_type, description in fault_types_to_test:
        print(f"\n{'='*60}")
        print(f"測試故障類型: {fault_type} ({description})")
        print(f"{'='*60}")
        
        # 載入STFT數據
        stft_data, loaded_fault_type = load_fault_stft_sample(
            fault_type=fault_type, 
            condition='4_0', 
            sample_index=0
        )
        
        if stft_data is not None:
            # 視覺化數據
            visualize_stft_before_sending(stft_data, fault_type)
            
            # 詢問是否要傳送
            user_input = input(f"是否要將 {fault_type} ({description}) 的STFT數據傳送到MCU？(y/n): ")
            
            if user_input.lower() == 'y':
                # 傳送到MCU並接收診斷結果
                print(f"🚀 開始傳送 {fault_type} 數據...")
                diagnosis_result = send_stft_to_mcu(stft_data, fault_type)
                
                # 分析結果
                is_correct = analyze_diagnosis_result(diagnosis_result, fault_type)
                
                # 記錄結果
                test_results.append({
                    'fault_type': fault_type,
                    'description': description,
                    'result': diagnosis_result,
                    'is_correct': is_correct
                })
                
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                # 等待用戶確認
                input("請確認MCU的LED狀態，然後按Enter繼續...")
                print(f"✅ 故障類型 {fault_type} 測試完成\n")
            else:
                print(f"⏭️  跳過 {fault_type} 的傳送")
        else:
            print(f"❌ 無法載入 {fault_type} 的數據")
    
    # 顯示測試總結
    print_test_summary(test_results, correct_count, total_count)

def print_test_summary(test_results, correct_count, total_count):
    """
    打印測試總結
    """
    print(f"\n{'='*80}")
    print(f"🎯 測試總結報告")
    print(f"{'='*80}")
    
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"📈 總體準確率: {correct_count}/{total_count} ({accuracy:.1f}%)")
    else:
        print("📈 未執行任何測試")
    
    print(f"\n📋 詳細結果:")
    for i, result in enumerate(test_results, 1):
        status_icon = "✅" if result['is_correct'] else "❌"
        print(f"   {i}. {status_icon} {result['fault_type']} ({result['description']})")
        if result['result']:
            print(f"      診斷: {result['result']['status']}, 信心度: {result['result']['confidence']}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    print("STFT故障診斷測試程式")
    print("此程式將載入真實的故障STFT數據並傳送到MCU進行測試")
    
    # 檢查串列埠
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    print("\n可用的串列埠:")
    for port in ports:
        print(f"  {port.device} - {port.description}")
    
    # 執行主程式
    main()












