import serial
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import sys

def create_progress_bar(current, total, prefix='Progress', suffix='Complete', 
                       decimals=1, bar_length=50):
    """
    創建進度條
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (current / float(total)))
    filled_length = int(round(bar_length * current / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))
    
    if current == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def send_stft_to_mcu_with_progress(stft_data, port='/dev/tty.usbserial-10'):
    """
    發送單個 STFT 樣本到 MCU 並接收診斷結果（帶進度條）
    """
    try:
        # 建立串列連接
        ser = serial.Serial(port, 115200, timeout=10)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # 數據預處理
        stft_flat = stft_data.flatten()
        #stft_normalized = (stft_flat - np.min(stft_flat)) / (np.max(stft_flat) - np.min(stft_flat))
        stft_normalized = (stft_flat - (-115.0041)) / (3.395 - (-115.0041))
        stft_data_uint8 = (stft_normalized * 255).astype(np.uint8)
        
        print(f"準備發送 {len(stft_data_uint8)} 個數據點")
        
        # 發送開始標記
        ser.write(bytes([0xFF]))
        ser.flush()
        print("已發送開始標記")
        
        # 初始化進度條
        create_progress_bar(0, len(stft_data_uint8), prefix='發送數據', suffix='完成')
        
        # 分批發送數據並顯示進度
        batch_size = 32
        total_sent = 0
        
        for i in range(0, len(stft_data_uint8), batch_size):
            batch = stft_data_uint8[i:i+batch_size]
            ser.write(batch)
            ser.flush()
            time.sleep(0.001)
            
            total_sent += len(batch)
            
            # 更新進度條
            create_progress_bar(total_sent, len(stft_data_uint8), 
                              prefix='發送數據', suffix='完成')
        
        # 發送結束標記
        ser.write(bytes([0xFE]))
        ser.flush()
        print("已發送結束標記")
        
        # 等待並接收診斷結果
        print("等待 MCU 診斷結果...")
        diagnosis_result = receive_diagnosis_result(ser, timeout=100)
        
        # 顯示結果
        if diagnosis_result and diagnosis_result['prediction'] != -1:
            print(f"✅ 診斷成功: {diagnosis_result['status']} (信心度: {diagnosis_result['confidence']:.4f})")
        else:
            print(f"❌ 診斷失敗或超時")
        
        ser.close()
        return diagnosis_result
        
    except Exception as e:
        print(f"❌ 發送樣本時發生錯誤: {e}")
        return None


def receive_diagnosis_result(ser, timeout=15):
    """
    接收 MCU 的診斷結果
    """
    start_time = time.time()
    response_buffer = ""
    
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            try:
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                response_buffer += data
                
                # 檢查診斷結果
                lines = response_buffer.split('\n')
                for line in lines:
                    line = line.strip()
                    
                    # 檢查 DIAGNOSIS 格式
                    if "DIAGNOSIS:" in line:
                        if "FAULT" in line:
                            try:
                                confidence_start = line.find("confidence:") + 11
                                confidence_end = line.find(")", confidence_start)
                                confidence = float(line[confidence_start:confidence_end])
                                return {'status': 'FAULT', 'confidence': confidence, 'prediction': 1}
                            except:
                                return {'status': 'FAULT', 'confidence': 0.5, 'prediction': 1}
                        elif "HEALTHY" in line:
                            try:
                                confidence_start = line.find("confidence:") + 11
                                confidence_end = line.find(")", confidence_start)
                                confidence = float(line[confidence_start:confidence_end])
                                return {'status': 'HEALTHY', 'confidence': confidence, 'prediction': 0}
                            except:
                                return {'status': 'HEALTHY', 'confidence': 0.5, 'prediction': 0}
                    
                    # 檢查 RESULT 格式
                    elif "RESULT:" in line:
                        try:
                            result_parts = line.replace("RESULT:", "").split(',')
                            diagnosis = result_parts[0].strip()
                            confidence = float(result_parts[1].strip())
                            
                            if "FAULT" in diagnosis:
                                return {'status': 'FAULT', 'confidence': confidence, 'prediction': 1}
                            else:
                                return {'status': 'HEALTHY', 'confidence': confidence, 'prediction': 0}
                        except:
                            return {'status': 'UNKNOWN', 'confidence': 0.0, 'prediction': -1}
            except Exception as e:
                print(f"解碼錯誤: {e}")
        
        time.sleep(0.1)
    
    return {'status': 'TIMEOUT', 'confidence': 0.0, 'prediction': -1}

def load_test_dataset():
    """
    載入測試集數據
    """
    try:
        X_test = np.load('X_test_stft.npy')
        y_test = np.load('y_test.npy')
        
        print(f"成功載入測試集：")
        print(f"  數據形狀: {X_test.shape}")
        print(f"  標籤形狀: {y_test.shape}")
        print(f"  健康樣本: {np.sum(y_test == 0)} 個")
        print(f"  故障樣本: {np.sum(y_test == 1)} 個")
        
        return X_test, y_test
    except FileNotFoundError:
        print("錯誤：找不到測試集檔案！")
        print("請確保以下檔案存在：")
        print("  - X_test_stft.npy")
        print("  - y_test.npy")
        return None, None

def plot_confusion_matrix(y_true, y_pred, accuracy):
    """
    繪製混淆矩陣
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Fault'], 
                yticklabels=['Healthy', 'Fault'])
    plt.title(f'MCU Inference Confusion Matrix\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('mcu_inference_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def batch_test_mcu_with_testset(num_samples=None, port='/dev/tty.usbserial-10'):
    """
    使用測試集進行批量 MCU 推論測試（帶進度顯示）
    """
    # 載入測試集
    X_test, y_test = load_test_dataset()
    if X_test is None:
        return None, None
    
    # 確定測試樣本數量
    if num_samples is None or num_samples > len(X_test):
        num_samples = len(X_test)
        print(f"將測試全部 {num_samples} 個樣本")
    else:
        print(f"將測試 {num_samples} 個樣本（從 {len(X_test)} 個中選取）")
    
    # 選擇測試樣本（保持平衡）
    if num_samples < len(X_test):
        healthy_indices = np.where(y_test == 0)[0]
        fault_indices = np.where(y_test == 1)[0]
        
        num_healthy = min(num_samples // 2, len(healthy_indices))
        num_fault = min(num_samples // 2, len(fault_indices))
        
        selected_healthy = np.random.choice(healthy_indices, num_healthy, replace=False)
        selected_fault = np.random.choice(fault_indices, num_fault, replace=False)
        
        test_indices = np.concatenate([selected_healthy, selected_fault])
        np.random.shuffle(test_indices)
    else:
        test_indices = np.arange(len(X_test))
    
    print(f"開始批量測試 {len(test_indices)} 個樣本...")
    print(f"健康樣本：{np.sum(y_test[test_indices] == 0)} 個")
    print(f"故障樣本：{np.sum(y_test[test_indices] == 1)} 個")
    
    # 執行測試
    y_true = []
    y_pred = []
    confidences = []
    failed_samples = 0
    
    for i, idx in enumerate(test_indices):
        print(f"\n{'='*60}")
        print(f"測試樣本 {i+1}/{len(test_indices)} (索引: {idx})")
        print(f"真實標籤: {'FAULT' if y_test[idx] == 1 else 'HEALTHY'}")
        print(f"{'='*60}")
        
        # 發送到 MCU（使用帶進度條的版本）
        result = send_stft_to_mcu_with_progress(X_test[idx], port)
        
        if result and result['prediction'] != -1:
            y_true.append(int(y_test[idx]))
            y_pred.append(result['prediction'])
            confidences.append(result['confidence'])
            
            is_correct = (result['prediction'] == y_test[idx])
            print(f"MCU 預測: {result['status']} (信心度: {result['confidence']:.4f})")
            print(f"結果: {'✅ 正確' if is_correct else '❌ 錯誤'}")
        else:
            failed_samples += 1
            print(f"❌ 推論失敗或超時")
        
        # 短暫延遲避免過載
        time.sleep(0.5)
    
    # 計算統計結果（保持原有邏輯）
    if len(y_pred) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n{'='*60}")
        print(f"MCU 批量測試結果")
        print(f"{'='*60}")
        print(f"總樣本數: {len(test_indices)}")
        print(f"成功推論: {len(y_pred)}")
        print(f"失敗樣本: {failed_samples}")
        print(f"準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"平均信心度: {np.mean(confidences):.4f}")
        
        # 生成分類報告
        print(f"\n分類報告:")
        class_names = ['Healthy', 'Fault']
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)
        
        # 繪製混淆矩陣
        cm = plot_confusion_matrix(y_true, y_pred, accuracy)
        
        # 保存詳細結果
        results = {
            'total_samples': len(test_indices),
            'successful_predictions': len(y_pred),
            'failed_samples': failed_samples,
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'confidences': confidences,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        np.save('mcu_testset_results.npy', results)
        print(f"\n詳細結果已保存至 'mcu_testset_results.npy'")
        
        return results, accuracy
    else:
        print("❌ 所有樣本推論都失敗了！")
        return None, None


def compare_with_pc_results():
    """
    比較 MCU 和 PC 端的推論結果（如果有的話）
    """
    try:
        # 載入 PC 端的交叉驗證結果
        pc_results = np.load('cross_validation_results.npy', allow_pickle=True).item()
        pc_accuracy = pc_results.get('test_accuracy', None)
        
        # 載入 MCU 測試結果
        mcu_results = np.load('mcu_testset_results.npy', allow_pickle=True).item()
        mcu_accuracy = mcu_results.get('accuracy', None)
        
        if pc_accuracy and mcu_accuracy:
            print(f"\n{'='*60}")
            print(f"PC vs MCU 性能比較")
            print(f"{'='*60}")
            print(f"PC 端測試準確率:  {pc_accuracy:.4f} ({pc_accuracy*100:.2f}%)")
            print(f"MCU 端測試準確率: {mcu_accuracy:.4f} ({mcu_accuracy*100:.2f}%)")
            print(f"準確率差異:       {abs(pc_accuracy - mcu_accuracy):.4f} ({abs(pc_accuracy - mcu_accuracy)*100:.2f}%)")
            
            if abs(pc_accuracy - mcu_accuracy) < 0.05:
                print("✅ MCU 和 PC 端性能相近")
            else:
                print("⚠️  MCU 和 PC 端性能有明顯差異，需要進一步優化")
        else:
            print("無法找到 PC 端結果進行比較")
            
    except FileNotFoundError:
        print("找不到比較檔案，跳過性能比較")

def main():
    """
    主函數 - 測試集批量推論
    """
    print("MCU 測試集批量推論程式")
    print("基於 STFT + CNN 的馬達故障診斷")
    print(f"{'='*60}")
    
    # 檢查串列埠
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    print("\n可用的串列埠:")
    for port in ports:
        print(f"  {port.device} - {port.description}")
    
    # 詢問測試參數
    try:
        num_samples = input("\n請輸入要測試的樣本數量 (按 Enter 測試全部): ")
        if num_samples.strip() == "":
            num_samples = None
        else:
            num_samples = int(num_samples) 
    except ValueError:
        num_samples = None
    
    port = '/dev/tty.usbserial-10'
    
    # 執行批量測試
    print(f"\n開始測試...")
    results, accuracy = batch_test_mcu_with_testset(num_samples, port)
    
    if results:
        # 比較 PC 和 MCU 性能
        compare_with_pc_results()
        
        print(f"\n{'='*60}")
        print("測試完成！生成的檔案：")
        print("  - mcu_testset_results.npy (詳細結果)")
        print("  - mcu_inference_confusion_matrix.png (混淆矩陣圖)")
        print(f"{'='*60}")
    else:
        print("測試失敗，請檢查 MCU 連接和設定。")

if __name__ == "__main__":
    main()
