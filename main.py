import pandas as pd
import torch
import torch.nn as nn
from  sklearn.preprocessing import OneHotEncoder,MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score



df=pd.read_csv('gender_classification_v7.csv')


scaler=MinMaxScaler()
encoder=OneHotEncoder(sparse_output=False)


# select columns
X=df.iloc[:,0:7]
y = LabelEncoder().fit_transform(df['gender'])

XScaled = scaler.fit_transform(X)


# encoded = pd.DataFrame(
#     encoder.fit_transform(df[['gender']] ),
#     columns=encoder.get_feature_names_out(['gender']),
#     index=df.index
# )





# encoded=pd.DataFrame(
#     encoder.fit_transform(y[['gender']]),
#     columns=encoder.get_feature_names_out(['gender']),
#     index=y.index
# )


X_train, X_test, y_train, y_test = train_test_split(XScaled,y,test_size=0.2,random_state=42)

X_train=torch.tensor(X_train,dtype=torch.float)
X_test=torch.tensor(X_test,dtype=torch.float)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)


class AnnClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputLayer=nn.Linear(in_features=7,out_features=128)
        self.hiddenLayer1=nn.Linear(in_features=128,out_features=64)
        self.hiddenLayer2=nn.Linear(in_features=64,out_features=32)
        self.outputLayer=nn.Linear(in_features=32,out_features=2)
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
        self.relu3=nn.ReLU()

    def forward(self,x):
        x=self.inputLayer(x)
        x=self.relu1(x)
        x=self.hiddenLayer1(x)
        x=self.relu2(x)
        x=self.hiddenLayer2(x)
        x=self.relu3(x)
        x=self.outputLayer(x)
        return x


model=AnnClassifier()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=5e-5)
epochs=1000
losses=[]
for epoch in range(epochs):
    y_pred=model(X_train)
    loss=criterion(y_pred,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%10==0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    losses.append(loss.item())


plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss_plot.png')  # This creates a file in your project folder
print("Plot saved as loss_plot.png")

with torch.no_grad():
    y_pred_logist=model(X_test)
    y_pred=torch.argmax(y_pred_logist,dim=1)

    acc=accuracy_score(y_test.numpy(),y_pred.numpy())
    print(f'Accuracy: {acc*100:.2f}%')


