# Face Anti Spoofing Detection
此專案為「工科海工專論」課程中之期末報告，主要參考論文為 "Searching Central Difference Convolutional Networks for Face Anti-Spoofing"
論文特點包含：
- 使用CDC特徵讀取
- 可抓出如lattice artifacts的細微特徵 (相片/影片的微小pixel)
- 使用 **深度圖depth map** 作為輸出
![image](https://github.com/user-attachments/assets/89c4b57b-c428-4fc3-bbf1-5ca37809d2fb)

## Train Result
底下為 OULU 與 SiW兩種測試集的訓練成果

### OULU dataset
![image](https://github.com/user-attachments/assets/3a84db08-5981-4e01-93fb-50ddbdd9fa9d)

### SiW dataset
![image](https://github.com/user-attachments/assets/5ee06c68-a601-4754-8d6b-b50a109f0119)

### Cross datasets
![image](https://github.com/user-attachments/assets/69c4a086-1653-4bfa-9cdd-2eef7ebce10d)

- SiW的真臉辨識率低、泛用性可能較差(也有可能是protocol選擇的問題)
- OULU較為穩定
