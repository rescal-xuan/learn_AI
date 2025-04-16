import torch  
from torch  import  nn  

class simplecnn(nn.Module):
    
    def __init__(self,num_class):
        super().__init__()
        self.num_class =num_class
        
        self.features =nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1), #  batch,3,224*224 -->batch,16,224*224
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # batch,16,224*224 --> batch,16,112*112
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),  # batch,16,112*112-->batch,32,112*112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # batch,32,112*112-->  batch,32,56*56
        )

        self.Linear =nn.Sequential(
            
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,num_class)
        )
        
    def forward(self,x):
        
        x =self.features(x)
        x= x.view(x.size(0),-1)
        x =self.Linear(x)
        
        return  x 
    
if  __name__  == "__main__":
    input_x = torch.randn(32,3,224,224)
    print(input_x.shape)
    model =simplecnn(num_class=4)
    out =model(input_x)
    print(out.shape)
    