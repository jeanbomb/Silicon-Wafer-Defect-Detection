# 矽晶圓缺陷檢測 | Silicon Wafer Defect Detection

本專案的目標是建立一個機器學習模型，來檢測矽晶圓圖片中的缺陷。這個專案使用的數據集來自 **WM811K 矽晶圓地圖數據集**，該數據集包含了缺陷和無缺陷的晶圓圖片。

The goal of this project is to build a machine learning model to detect defects in silicon wafer images. The dataset used in this project is the **WM811K Silicon Wafer Map Dataset**, which includes images of wafers with defects and without defects.

## 專案概覽 | Project Overview

本專案的目標是將晶圓圖片分類為兩個類別：

- **有缺陷**（例如：有刮痕或裂紋的晶圓）
- **無缺陷**（完美的晶圓，無明顯缺陷）

This project aims to classify silicon wafer images into two categories:
- **Defective** (e.g., wafers with scratches or cracks)
- **Non-defective** (perfect wafers without obvious defects)

使用了卷積神經網絡（CNN）結合 TensorFlow 和 Keras 來訓練模型，並在測試數據集上進行評估。

We used a Convolutional Neural Network (CNN) with TensorFlow and Keras to train the model and evaluate it on the test dataset.

## 安裝需求 | Installation Requirements

要運行此專案，你需要安裝以下的 Python 庫：

To run this project, you need to install the following Python libraries:

```bash
pip install tensorflow
pip install numpy
pip install matplotlib
pip install opencv-python
```

## 數據集 | Dataset

本專案使用的數據集來自 **WM811K 矽晶圓地圖數據集**，可以在 Kaggle 上找到。該數據集包含了顯示晶圓表面有缺陷（如刮痕、裂紋等）和無缺陷的圖片。你可以從以下鏈接下載：

- [WM811K 矽晶圓地圖數據集](https://www.kaggle.com/datasets/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image)

### 資料夾結構 | Folder Structure

數據集被分為多個資料夾，每個資料夾包含不同類型的缺陷或無缺陷的晶圓圖片：

The dataset is divided into several folders, each containing different types of defects or non-defective wafers:

- **Scratch**: 有刮痕的晶圓圖片
- **none**: 無缺陷的晶圓圖片（理想晶圓）

For example:

```plaintext
- WM811k_Dataset
  - Scratch
  - none
```

## 數據預處理 | Data Preprocessing

1. 圖片被調整為一致的尺寸（例如：64x64 像素），以確保在模型訓練過程中具有一致性。
2. 圖片被標註為 **有缺陷**（例如：刮痕）或 **無缺陷**（理想晶圓圖片）。
3. 圖片數據進行了正規化，將像素值縮放到 0 到 1 之間。

1. Images are resized to a consistent size (e.g., 64x64 pixels) to ensure consistency during model training.
2. Images are labeled as **Defective** (e.g., scratches) or **Non-defective** (ideal wafer images).
3. Image data is normalized to scale pixel values between 0 and 1.

## 模型架構 | Model Architecture

使用 Keras 建立了一個卷積神經網絡（CNN），其架構如下：

We built a Convolutional Neural Network (CNN) with Keras, and its architecture is as follows:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # 2 類別：有缺陷和無缺陷
])
```

## 訓練與評估 | Training and Evaluation

使用 **80%** 的數據來訓練模型，並使用 **20%** 進行驗證。訓練過程中使用了 **Adam** 優化器和 **交叉熵損失函數**。模型訓練了 **50** 個世代（epochs）。

We used **80%** of the data for training the model and **20%** for validation. During training, we used the **Adam** optimizer and **cross-entropy loss function**. The model was trained for **50** epochs.

訓練後，在測試集上評估模型，結果顯示測試準確度達到了 **95%**。

After training, we evaluated the model on the test set, and the result showed that the test accuracy reached **95%**.

### 訓練過程 | Training Process

```python
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```

### 損失和準確度圖表 | Loss and Accuracy Plots

在訓練過程中，繪製了訓練集和驗證集的損失與準確度曲線，來分析模型的訓練過程。

During the training process, we plotted the loss and accuracy curves for the training and validation sets to analyze the model's training process.

```python
plt.plot(history.history['accuracy'], label='訓練準確度')
plt.plot(history.history['val_accuracy'], label='驗證準確度')
plt.title('訓練與驗證準確度')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='驗證損失')
plt.title('訓練與驗證損失')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## 下一步 | Next Steps

儘管模型的準確度已達 **95%**，但仍有一些方法可以進一步提高其表現，包含：

- **數據增強**：使用旋轉、翻轉、縮放等變換來增加訓練數據。
- **模型優化**：嘗試更深的模型架構，或使用預訓練模型（如 ResNet、Inception）進行微調。
- **超參數調整**：嘗試不同的學習率、批次大小或優化器設定。
- **微調**：使用預訓練模型並進行微調，以提高在特定任務上的表現。

Although the model accuracy has reached **95%**, there are still ways to improve its performance, including:
- **Data Augmentation**: Use transformations such as rotation, flipping, and scaling to increase the training data.
- **Model Optimization**: Try deeper architectures or fine-tune pre-trained models (e.g., ResNet, Inception).
- **Hyperparameter Tuning**: Try different learning rates, batch sizes, or optimizer settings.
- **Fine-tuning**: Use pre-trained models and fine-tune them to improve performance on the specific task.

## 結論 | Conclusion

這個專案展示了深度學習在晶圓缺陷檢測中的應用。模型已經達到 **95%** 的測試準確度，這表明它在此任務中表現出色。未來，可以進一步優化模型，並將其應用於半導體製造業中的實際場景中。

This project demonstrates the application of deep learning in silicon wafer defect detection. The model has achieved **95%** test accuracy, indicating its excellent performance on this task. In the future, we can further optimize the model and apply it to real-world scenarios in the semiconductor manufacturing industry.
