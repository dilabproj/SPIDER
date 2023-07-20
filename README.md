## 說明文件

### 本說明文件是關於如何在bi-branch self-supervised ECG架構下訓練的ECG encoder，並以該ECG encoder對心電圖訊號進行轉換，以用於其他ECG相關應用。
### 本研究使用三個ECG dataset進行模型訓練，分別為PTBXL、Chapman以及CPSC，因此會有三個對應的訓練模型。資料集皆為開源資料集，本研究所使用的資料來源如下：
* PTBXL：https://physionet.org/content/ptb-xl/1.0.3/
* Chapman：https://physionet.org/content/ecg-arrhythmia/1.0.0/
* CPSC：http://2018.icbeb.org/Challenge.html

使用步驟如下：
* 訓練ECG encoder
    1. 下載欲使用的心電圖的資料集
    2. 以conda依requirement.txt建立虛擬環境
    3. 在run_experiment.py設定資料集路徑、GPU ID、儲存位置，或是其他欲更改的模型設定
    4. 以python執行run_experiment.py進行訓練
* 使用ECG encoder
    1. 設定正確的模型參數及模型儲存位置
    2. 將欲轉換的ECG資料包裝成torch.utils.data.DataLoader類別，輸出維度維(BatchSize, Channels, Length)的資料。
    3. 透過model.encode(DataLoader)進行特徵轉換

## Acknowledgement

This project was supported by Ministry of Science and Technology, Taiwan, under grant no. <font color=#FF0000>MOST ???.</font> 
