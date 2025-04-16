import torch
from  torch import nn,optim
from model import Transformer
from  dataset import *
from tqdm import tqdm
from transformers   import  AutoTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 2. 加载tokenizer
tokenzier = AutoTokenizer.from_pretrained('./model_/gpt2_chinese')

# 3. 数据路径和参数
train_data_path = './data/train.txt'
test_data_path = './data/test.txt'
max_len = 64  # 假设的 max_len
batch_size = 4 # 你可以调整 batch_size
def collate_fn(batch):
    
    src=batch[0][0]
    tgt =batch[0][1]
    src =tokenzier(src, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")['input_ids']
    tgt =tokenzier(tgt, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")['input_ids']
    tgt_labels=tgt
    
    return  src,tgt,tgt_labels
# 4. 加载数据集
train_data = transDataSet(tokenzier, train_data_path, max_len)  # 替换为你的 transDataSet 类
trainloader = DataLoader(train_data, batch_size=batch_size,collate_fn=collate_fn,shuffle=True)

test_data = transDataSet(tokenzier, test_data_path, max_len) # 替换为你的 transDataSet 类
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 5. 模型参数
src_vocab_size = len(tokenzier)
tgt_vocab_size = len(tokenzier)
heads = 8
d_model = 512
d_ff = 1024
num_layer = 6
pad_idx = tokenzier.pad_token_id

# 6. 初始化模型
model = Transformer(src_vocab_size, tgt_vocab_size,heads,d_model, d_ff, num_layer, max_seq_len=max_len).to(device) 
# 7. 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fun = nn.CrossEntropyLoss(ignore_index=pad_idx)



  

# 8. 训练参数
epochs = 200
best_acc = 0.0  # 初始化最佳准确率

# 9. 训练循环
for epoch in tqdm(range(epochs), desc="Epochs"):
    model.train()
    for index, (src, tgt, tgt_labels) in enumerate(trainloader):
        src, tgt, tgt_labels = src.to(device), tgt.to(device), tgt_labels.to(device)

        

        output = model(src, tgt)  # 模型前向传播
        pred_ = torch.argmax(output, dim=-1)  # 获取预测的 token
        acc = (pred_ == tgt_labels).float().mean()  # 计算准确率

        output_ = output.reshape(-1, output.shape[-1])  # 将输出展开为 (batch_size * seq_len, vocab_size)
        tgt_label_ = tgt_labels.reshape(-1)  # 将目标标签展开为 (batch_size * seq_len)
        train_loss = loss_fun(output_, tgt_label_)  # 计算损失
        optimizer.zero_grad()  # 清零梯度
        train_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if index % 100 == 0:
            print(f"Epoch: {epoch} iter {index / len(trainloader):.2f} train loss: {train_loss.item():.4f}  acc: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './model.pt')
            # print(f"保存模型在第{epoch}轮,准确率为{best_acc:.3f}")
    # 10. 测试循环
    model.eval()
    with torch.no_grad():
        test_acc_sum = 0.0
        test_loss_sum = 0.0
        num_batches = len(testloader)
        for index, (src, tgt, tgt_label) in enumerate(testloader):
            src, tgt, tgt_label = src.to(device), tgt.to(device), tgt_label.to(device)
            output = model(src, tgt)
            pred_ = torch.argmax(output, dim=-1)
            acc = (pred_ == tgt_label).float().mean()
            output_ = output.reshape(-1, output.shape[-1])
            tgt_label_ = tgt_label.reshape(-1)
            test_loss = loss_fun(output_, tgt_label_)
            test_acc_sum += acc
            test_loss_sum += test_loss

            if index % 100 == 0:
                print(f"Epoch: {epoch} iter {index / len(testloader):.2f} test loss: {test_loss.item():.4f}  acc: {acc:.3f}")

        avg_test_acc = test_acc_sum / num_batches
        avg_test_loss = test_loss_sum / num_batches
        print(f"Epoch {epoch}: Average Test Loss: {avg_test_loss:.4f}, Average Test Accuracy: {avg_test_acc:.3f}")

print("Training finished.")
