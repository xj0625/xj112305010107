import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from PIL import Image
import random

# 加载数据
def load_data():
    train_df = pd.read_csv('train.csv')
    X_train = train_df.drop('label', axis=1).values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.int64)
    
    test_df = pd.read_csv('test.csv')
    X_test = test_df.values.astype(np.float32)
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    
    return X_train, y_train, X_test

# 增强的数据增强
class DataAugmentation:
    @staticmethod
    def shift(image, max_shift=3):
        batch_size, _, height, width = image.size()
        
        shift_x = torch.randint(-max_shift, max_shift + 1, (batch_size,))
        shift_y = torch.randint(-max_shift, max_shift + 1, (batch_size,))
        
        shifted = torch.zeros_like(image)
        for i in range(batch_size):
            sx, sy = shift_x[i], shift_y[i]
            x_start = max(0, sx)
            x_end = min(width, width + sx)
            y_start = max(0, sy)
            y_end = min(height, height + sy)
            
            src_x_start = max(0, -sx)
            src_x_end = min(width, width - sx)
            src_y_start = max(0, -sy)
            src_y_end = min(height, height - sy)
            
            shifted[i, :, y_start:y_end, x_start:x_end] = image[i, :, src_y_start:src_y_end, src_x_start:src_x_end]
        
        return shifted
    
    @staticmethod
    def rotate(image, max_angle=15):
        batch_size, _, height, width = image.size()
        rotated = torch.zeros_like(image)
        
        for i in range(batch_size):
            angle = random.uniform(-max_angle, max_angle)
            img = image[i, 0].numpy()
            
            # 简单旋转实现
            from scipy.ndimage import rotate
            rotated_img = rotate(img, angle, reshape=False, order=1, mode='constant', cval=0.0)
            rotated[i, 0] = torch.tensor(rotated_img)
        
        return rotated
    
    @staticmethod
    def zoom(image, scale_range=(0.9, 1.1)):
        batch_size, _, height, width = image.size()
        zoomed = torch.zeros_like(image)
        
        for i in range(batch_size):
            scale = random.uniform(*scale_range)
            img = image[i, 0].numpy()
            
            # 简单缩放实现
            from scipy.ndimage import zoom as scipy_zoom
            zoomed_img = scipy_zoom(img, scale, order=1, mode='constant', cval=0.0)
            
            # 调整回28x28
            zoomed_img_pil = Image.fromarray(zoomed_img.astype(np.float32))
            zoomed_img_pil = zoomed_img_pil.resize((28, 28), Image.LANCZOS)
            zoomed_img = np.array(zoomed_img_pil)
            
            zoomed[i, 0] = torch.tensor(zoomed_img)
        
        return zoomed

# 构建增强的CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # 第二层
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # 第三层
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(0.3)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv_layers(x)
        x = x.view(-1, 256)
        x = self.fc_layers(x)
        return x

# 训练模型（带OneCycleLR和增强的数据增强）
def train_model(model, X_train, y_train, val_ratio=0.1, epochs=30, batch_size=512, learning_rate=0.003, patience=8):
    dataset = TensorDataset(X_train, y_train)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 使用OneCycleLR学习率调度器
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0
    )
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    augmenter = DataAugmentation()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # 数据增强 - 随机选择一种或多种增强方式
            inputs_reshaped = inputs.view(-1, 1, 28, 28)
            inputs_augmented = inputs_reshaped.clone()
            
            # 随机应用数据增强
            if random.random() < 0.7:
                choice = random.randint(0, 2)
                if choice == 0:
                    inputs_augmented = augmenter.shift(inputs_reshaped, max_shift=3)
                elif choice == 1:
                    inputs_augmented = augmenter.rotate(inputs_reshaped, max_angle=12)
                elif choice == 2:
                    inputs_augmented = augmenter.zoom(inputs_reshaped, scale_range=(0.9, 1.1))
            
            inputs_augmented = inputs_augmented.view(-1, 784)
            
            outputs = model(inputs_augmented)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Epoch {epoch+1}/{epochs}, Best model saved! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.2f}% at epoch {best_epoch+1}')
                break
        
        model.train()
        
        if (epoch + 1) % 3 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    model.load_state_dict(torch.load('best_model.pth'))
    print(f'Training completed. Loaded best model from epoch {best_epoch+1} with val acc: {best_val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

# 绘制loss曲线
def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs (AdamW + OneCycleLR)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    print('Loss curve saved as: loss_curve.png')

# 绘制准确率曲线
def plot_acc_curve(train_accs, val_accs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy Over Epochs (AdamW + OneCycleLR)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('acc_curve.png', dpi=300, bbox_inches='tight')
    print('Accuracy curve saved as: acc_curve.png')

# TTA测试时增强
def test_with_tta(model, X_test, num_augments=3):
    model.eval()
    augmenter = DataAugmentation()
    all_probs = []
    
    with torch.no_grad():
        # 原始预测
        outputs = model(X_test)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs)
        
        # TTA增强
        for _ in range(num_augments):
            X_test_reshaped = X_test.view(-1, 1, 28, 28)
            X_test_augmented = X_test_reshaped.clone()
            
            choice = random.randint(0, 2)
            if choice == 0:
                X_test_augmented = augmenter.shift(X_test_reshaped, max_shift=3)
            elif choice == 1:
                X_test_augmented = augmenter.rotate(X_test_reshaped, max_angle=12)
            elif choice == 2:
                X_test_augmented = augmenter.zoom(X_test_reshaped, scale_range=(0.9, 1.1))
            
            X_test_augmented = X_test_augmented.view(-1, 784)
            outputs = model(X_test_augmented)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
        
        # 平均所有预测
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        _, predicted = torch.max(avg_probs, 1)
    
    return predicted

# 测试模型并生成提交文件
def test_model(model, X_test):
    print('开始TTA测试...')
    predicted = test_with_tta(model, X_test, num_augments=3)
    
    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted) + 1),
        'Label': predicted.numpy()
    })
    submission.to_csv('sample_submission.csv', index=False)
    print('提交文件已生成: sample_submission.csv')

# 主函数
def main():
    X_train, y_train, X_test = load_data()
    print(f'训练数据形状: {X_train.shape}')
    print(f'测试数据形状: {X_test.shape}')
    
    model = CNN()
    print('模型结构:')
    print(model)
    
    print('开始训练模型（AdamW + OneCycleLR + 增强数据增强）...')
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, 
        X_train, 
        y_train,
        epochs=30,
        batch_size=512,
        learning_rate=0.003,
        patience=8
    )
    
    print('绘制loss曲线...')
    plot_loss_curve(train_losses, val_losses)
    
    print('绘制准确率曲线...')
    plot_acc_curve(train_accs, val_accs)
    
    print('开始测试模型（TTA）...')
    test_model(model, X_test)

if __name__ == '__main__':
    main()
