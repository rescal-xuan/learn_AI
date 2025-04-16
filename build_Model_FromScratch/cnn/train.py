import  torch 
from torch  import nn,optim
from  torch.utils.data  import DataLoader
from  torchvision  import datasets,transforms
import os
from  tqdm  import tqdm 
from model  import simplecnn
batch_size =64
device ='cuda' if  torch.cuda.is_available()  else 'cpu'
epochs =10
model =simplecnn(num_class=2).to(device)
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

traindata =datasets.ImageFolder(root=os.path.join(r'dataset/dogvscat','train'),transform=data_transforms['train'])
trainloader =DataLoader(traindata,batch_size=batch_size,shuffle=True)

testdata =datasets.ImageFolder(root=os.path.join(r'dataset/dogvscat','test'),transform=data_transforms['test'])
testloader =DataLoader(testdata,batch_size=batch_size,shuffle=False)

def train(model, train_loader, optimizer, loss_function, device, epochs):
    model.train() 
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    best_acc=0
    for epoch in range(epochs):
        for images, labels  in tqdm(train_loader,desc=f"epoch: {epoch+1}/{epochs}",unit="batch"):
            images, labels = images.to(device), labels.to(device)
            
        
            outputs = model(images)
            loss = loss_function(outputs, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            
            
            # print(f'Epoch [{epoch+1}/{epochs}],  Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        acc = correct_predictions / total_samples
        print(f'Epoch {epoch+1} Train: Average Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}')
        evaluate(model, testloader, loss_function, device)
        if acc >best_acc :
            best_acc =acc
            torch.save(model.state_dict(),'./model_pt/model.pt')
    
def evaluate(model, test_loader, loss_function, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
   
    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
          
            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(test_loader)
        acc = correct_predictions / total_samples
        print(f'Test: Average Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')
        
            
    
if  __name__ =="__main__": 
    train_loss, train_acc = train(model, trainloader, optimizer, loss_function, device, epochs)
    test_loss, test_acc = evaluate(model, testloader, loss_function, device)
            
