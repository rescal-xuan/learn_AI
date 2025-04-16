import  torch 
from torch  import nn,optim
from  torch.utils.data  import DataLoader
from  torchvision  import datasets,transforms
import os
from  tqdm  import tqdm 
from model1  import ViTransformer
batch_size =64
device ='cuda' if  torch.cuda.is_available()  else 'cpu'
epochs =10
patch_size=16
d_model = 512

model = ViTransformer(d_model, 224,patch_size,10,3).to(device)
optimizer =optim.Adam(model.parameters(),lr=1e-3)
loss_function =nn.CrossEntropyLoss()
data_transforms ={
    'train':transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.52,0.52,0.52))]
    ),
    'test':transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.52,0.52,0.52))]
    )
    
}
data_dir = r'D:\learn_AI\build_Model_FromScratch\ViT\dogvscat'
pt_dir = './model_pt'

if not pt_dir or not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
train_data = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
trainloader =DataLoader(train_data,batch_size=batch_size,shuffle=True)

test_data = datasets.ImageFolder(root=test_dir, transform=data_transforms['test'])
testloader =DataLoader(test_data,batch_size=batch_size,shuffle=False)

def train(model, train_loader, optimizer, loss_function, device, epochs, testloader):
    """
    训练模型函数。

    Args:
        model (torch.nn.Module): 要训练的 PyTorch 模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器（例如 Adam, SGD）。
        loss_function (torch.nn.Module): 损失函数（例如交叉熵损失）。
        device (str):  'cuda' (如果可用) 或者 'cpu'，指定设备。
        epochs (int): 训练的总轮数。
        testloader (torch.utils.data.DataLoader): 测试数据加载器.

    Returns:
        tuple: (average_loss, accuracy) 训练的平均损失和准确率。
    """
    model.train()  # 设置模型为训练模式 (启用 dropout, batchnorm 等)
    best_acc = 0.0  # 初始化最佳准确率
    final_avg_loss = 0.0
    final_acc = 0.0

    for epoch in range(epochs):
        total_loss = 0.0  # 重置每个epoch的总损失
        correct_predictions = 0
        total_samples = 0

        # 使用 tqdm 显示训练进度
        for images, labels in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)  # 将数据移动到指定设备

            # 前向传播
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型参数

            # 统计指标
            total_loss += loss.item()  # 累加损失值
            _, predicted_labels = torch.max(outputs, 1)  # 获取预测标签
            correct_predictions += (predicted_labels == labels).sum().item()  # 统计正确预测的数量
            total_samples += labels.size(0)  # 统计样本总数

        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        acc = correct_predictions / total_samples
        final_avg_loss = avg_loss
        final_acc = acc

        # 打印训练信息
        print(f"Epoch {epoch+1} Train: Average Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

        # 在测试集上评估模型
        evaluate(model, testloader, loss_function, device)

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './model_pt/model.pt')
            print(f"保存最佳模型，准确率: {best_acc:.4f}")

    return final_avg_loss, final_acc


def evaluate(model, test_loader, loss_function, device):
    """
    评估模型函数。

    Args:
        model (torch.nn.Module): 要评估的 PyTorch 模型。
        test_loader (torch.utils.data.DataLoader): 测试数据加载器。
        loss_function (torch.nn.Module): 损失函数。
        device (str): 'cuda' (如果可用) 或者 'cpu'。
    """
    model.eval()  # 设置模型为评估模式 (禁用 dropout, batchnorm 等)
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 禁用梯度计算，加速评估过程，节省内存
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # 统计指标
            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    # 计算平均损失和准确率
    avg_loss = total_loss / len(test_loader)
    acc = correct_predictions / total_samples

    # 打印评估信息
    print(f"Test: Average Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        
            
    
if  __name__ =="__main__": 
    train_loss, train_acc = train(model, trainloader, optimizer, loss_function, device, epochs,testloader)
    # test_loss, test_acc = evaluate(model, testloader, loss_function, device)
            
