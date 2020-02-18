import torch
import torch.nn as nn

#二分类交叉熵
#>1.Sigmoid激活+BCELoss
f=nn.Sigmoid()
E=nn.BCELoss()#reduction采用sum and mean

X=torch.randn(3,2)
print('X is:\n',X)
t=torch.empty(3,2).random_(0,2)
print('\nt is:\n',t)
out=E(f(X),t)#以element-wise计算loss
print('\nout is:\n',out)

#>2.直接使用带激活的BCEWithLogitsLoss
E_combine=nn.BCEWithLogitsLoss()
out=E_combine(X,t)
print('\nout is:\n',out)


#多分类交叉熵
##>1.LogSoftmax激活+NLLLoss
f=nn.LogSoftmax(dim=1)
E=nn.NLLLoss()#sum and mean

X=torch.randn(5,3)
t=torch.empty(5,1,dtype=torch.long).random_(0,3).squeeze(1) #只能是单一维度(5,)
#t=torch.empty(5,1,dtype=torch.long).random_(0,3) #shape(5,1)不可以

print('X is:\n',X)
print('\nt is:\n',t)
out=E(f(X),t)
print('\nout is:\n',out)

##>2.直接使用带激活的CrossEntropyLoss
E_combine=nn.CrossEntropyLoss()
out=E_combine(X,t)
print('\nout is:\n',out)
