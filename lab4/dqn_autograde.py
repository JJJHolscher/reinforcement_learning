import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
        
        raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
        print("x", x)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
        print("x", x.shape)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        a = self.l1(x)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        a = self.l1(x)
        print(a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        a = self.l1(x)
        print("a", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        a = self.l1(x)
        print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        output_relu = nn.Relu(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        output_ReLu = nn.Relu(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        output_relu = nn.ReLu(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        output_relu = Torch.ReLu(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        output_relu = nn.ReLU(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer)
        output_relu = nn.ReLU(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(type(output_firt_lin_layer))
        output_relu = nn.ReLU(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = nn.ReLU(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = nn.ReLU(output_firt_lin_layer)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = nn.ReLU(output_firt_lin_layer)
        print("out relu", output_relu)
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = nn.ReLU(output_firt_lin_layer)
        print("out relu", output_relu)
        print("##")
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = nn.ReLU(output_firt_lin_layer)
        print("out relu", output_relu)
#         print("##")
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = nn.ReLU(output_firt_lin_layer)
#         print("out relu", output_relu)
#         print("##")
        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
#         output_firt_lin_layer = self.l1(x)
    
#         nn.
#         print(output_firt_lin_layer.shape)
        output_relu = nn.ReLU(self.l1(x))

        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer)
    
#         nn.
#         print(output_firt_lin_layer.shape)
#         output_relu = nn.ReLU(self.l1(x))

        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
    
#         nn.
#         print(output_firt_lin_layer.shape)
#         output_relu = nn.ReLU(self.l1(x))

        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        print(type(output_firt_lin_layer))
    
#         nn.
#         print(output_firt_lin_layer.shape)
#         output_relu = nn.ReLU(self.l1(x))

        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        r = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        print(type(output_firt_lin_layer))

        output_relu = r(self.l1(x)

        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        r = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2
#         print("x", x.shape)
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        print(type(output_firt_lin_layer))

        output_relu = r(self.l1(x)
        b = self.l2(output_relu)

        return
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        r = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2

    
        output_firt_lin_layer = self.l1(x)
        output_relu = r(self.l1(x))
        b = self.l2(output_relu)

        return
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        r = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2

    
        output_firt_lin_layer = self.l1(x)
        output_relu = r(self.l1(x))
        b = self.l2(output_relu)

        return b
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2

    
        output_firt_lin_layer = self.l1(x)
        output_relu = relu(self.l1(x))
        b = self.l2(output_relu)

        return b
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2

    
        output_firt_lin_layer = self.l1(x)
        output_relu = relu(self.l1(x))
        b = self.l2(output_relu)

        return b
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2

    
        output_firt_lin_layer = self.l1(x)
        output_relu = F.relu(self.l1(x))
        b = self.l2(output_relu)

        return b
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2

    
        output_firt_lin_layer = self.l1(x)
        output_relu = F.relu(self.l1(x))
        b = self.l2(output_relu)

        return b
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2

    
        output_firt_lin_layer = self.l1(x)
        output_relu = F.relu(self.l1(x))
        b = self.l2(output_relu)

        return b
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
        # doe self.l1
        # doe relu nu
        # doe self.l2

    
        output_firt_lin_layer = self.l1(x)
        output_relu = F.relu(output_firt_lin_layer)
        b = self.l2(output_relu)

        return b
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
    
        output_firt_lin_layer = self.l1(x)
        output_relu = F.relu(output_firt_lin_layer)


        return self.l2(output_relu)
        
#         print("aaapi", a)
#         raise NotImplementedError

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
    
        output_firt_lin_layer = self.l1(x)
        output_relu = F.relu(output_firt_lin_layer)


        return self.l2(output_relu)

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
    
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = F.relu(output_firt_lin_layer)


        return self.l2(output_relu)

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
    
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = F.relu(output_firt_lin_layer)
        print(output.relu.shape)


        return self.l2(output_relu)

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
    
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = F.relu(output_firt_lin_layer)
        print(output_relu.shape)


        return self.l2(output_relu)

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        relu = nn.ReLU()
    
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = F.relu(output_firt_lin_layer)
        end =  self.l2(output_relu)
        print(end.shape)

        return end

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        print("transition", transition)
        # YOUR CODE HERE
#         raise NotImplementedError

    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
#         print("transition", transition)
        # YOUR CODE HERE
#         raise NotImplementedError

    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
#         print("transition", transition)
        # YOUR CODE HERE
        raise NotImplementedError

    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        print("transition", transition)
        # YOUR CODE HERE
        raise NotImplementedError

    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        memory.append(transition)
        print(memory)
        # YOUR CODE HERE
#         raise NotImplementedError

    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        memory.append(transition)
        print("MEEMMM", memory)
        # YOUR CODE HERE
#         raise NotImplementedError

    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        memory.append(transition)
        print("MEEMMM", memory)


    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        print("MEEMMM", memory)


    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        print("MEEMMM", memory)


    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        print("MEEMMM", self.memory)


    def sample(self, batch_size):
        # YOUR CODE HERE
        raise NotImplementedError

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        print("MEEMMM", self.memory)


    def sample(self, batch_size):
        random.sample(self.memory)

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        print("MEEMMM", self.memory)


    def sample(self, batch_size):
        random.sample(self.memory, 1)

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        print("MEEMMM", self.memory)

    def sample(self, batch_size):
        random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    # YOUR CODE HERE
    raise NotImplementedError
    return epsilon

def get_epsilon(it):
    print(it)
    # YOUR CODE HERE
    raise NotImplementedError
    return epsilon

def get_epsilon(it):
    print(it)
    # YOUR CODE HERE
#     raise NotImplementedError
    return epsilon

def get_epsilon(it):
    print(it)
    # YOUR CODE HERE
#     raise NotImplementedError
    return None

def get_epsilon(it):
    slope = -0.00095
    
    epsilon = it * slope
    print(epsilon)
    # YOUR CODE HERE
#     raise NotImplementedError
    return None

def get_epsilon(it):
    slope = -0.00095
    
    epsilon = 1 - (it * slope)
    print(epsilon)
    # YOUR CODE HERE
#     raise NotImplementedError
    return None

def get_epsilon(it):
    slope = -0.00095
    
    epsilon = 1 + (it * slope)
    print(epsilon)
    # YOUR CODE HERE
#     raise NotImplementedError
    return None

def get_epsilon(it):
    
    if it < 1000:
        slope = -0.00095

        epsilon = 1 + (it * slope)

        
    else:
        epsilon = 0.05
    return epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        raise NotImplementedError
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        print(self)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        print(self.Q)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        print(self.epsilon)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        print(self.epsilon)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        print(self.epsilon)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        print(self.epsilon)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#         print(self.epsilon)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
    print("obs", obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
    prediction = Q_net(obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
    print(self.Q)
#     prediction = Q_net(obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
        print(self.Q)
#     prediction = Q_net(obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
        print(self.Q)
    prediction = self.Q(obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        prediction = self.Q(obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        prediction = self.Q.forward(obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        print("JAAJAJA", x)
        relu = nn.ReLU()
    
        output_firt_lin_layer = self.l1(x)
        print(output_firt_lin_layer.shape)
        output_relu = F.relu(output_firt_lin_layer)
        end =  self.l2(output_relu)
        print(end.shape)

        return end

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs)
        print(type(obs))
        prediction = self.Q.forward(obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print(prediction)
        print(prediction.shape)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        print(prediction.shape)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        print(prediction.shape)
        print(self.epsilon)
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        print(prediction.shape)

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        print(prediction.shape)

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        print(prediction.shape)

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        print(type(prediction))

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.to_numpy()
        print(type(prediction))

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))
        print(prediction)

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))
        print(prediction)
        coin = np.random.uniform
        print()
        
        # trek een random nummer 0 - 1
        # als die onder 0.95 is dan doe je de hoogste waarde
        
        # als die erboven is dan doe je een random value

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))
        print(prediction)
        coin = np.random.uniform
        print(coin)
        
        # trek een random nummer 0 - 1
        # als die onder 0.95 is dan doe je de hoogste waarde
        
        # als die erboven is dan doe je een random value

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))
        print(prediction)
        coin = np.random.uniform()
        print(coin)
        
        # trek een random nummer 0 - 1
        # als die onder 0.95 is dan doe je de hoogste waarde
        
        # als die erboven is dan doe je een random value

        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))
        print(prediction)
        coin = np.random.uniform()
        print(coin)
        
        if coin <= 0.95:
            action = np.argmax(prediction)
            print(action)
        else:
            action = np.random()
        
        # trek een random nummer 0 - 1
        # als die onder 0.95 is dan doe je de hoogste waarde
        
        # als die erboven is dan doe je een random value

        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))
        print(prediction)
        coin = np.random.uniform()
        print(coin)
        
        if coin <= 0.95:
            action = np.argmax(prediction)
            print("AA", action)
        else:
            action = np.random()
        
        # trek een random nummer 0 - 1
        # als die onder 0.95 is dan doe je de hoogste waarde
        
        # als die erboven is dan doe je een random value

        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)
        print(type(obs))
        obs = torch.tensor(obs).float()
        print(type(obs))
        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()
        print(type(prediction))
        print(prediction)
        coin = np.random.uniform()
        print(coin)
        
        print(prediction)
        
        if coin <= 0.95:
            action = np.argmax(prediction)
            print("AA", action)
        else:
            action = np.random(low = 0, high = len(prediction) - 1)
        
        # trek een random nummer 0 - 1
        # als die onder 0.95 is dan doe je de hoogste waarde
        
        # als die erboven is dan doe je een random value

        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)

        obs = torch.tensor(obs).float()

        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()

        coin = np.random.uniform()
        print(coin)
        
        print(prediction)
        print("len", len(prediction))
        
        if coin <= 0.95:
            action = np.argmax(prediction)
            print("AA", action)
        else:
            action = np.random(low = 0, high = len(prediction) - 1)
        
        # trek een random nummer 0 - 1
        # als die onder 0.95 is dan doe je de hoogste waarde
        
        # als die erboven is dan doe je een random value

        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)

        obs = torch.tensor(obs).float()

        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()

        coin = np.random.uniform()
        print(coin)
        
        print(prediction)
        print("len", len(prediction))
        
        if coin <= 0.95:
            action = np.argmax(prediction)
            print("AA", action)
        else:
            action = np.random.uniform(low = 0, high = len(prediction) - 1)
        
        # trek een random nummer 0 - 1
        # als die onder 0.95 is dan doe je de hoogste waarde
        
        # als die erboven is dan doe je een random value

        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)

        obs = torch.tensor(obs).float()

        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()

        coin = np.random.uniform()
        
        # take greedy action
        if coin <= 1 - self.epsilon:
            action = np.argmax(prediction)
            
        #else select a random action
        else:
            action = np.random.uniform(low = 0, high = len(prediction) - 1)


        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
#     print("obs", obs)
#         print(self.Q)

        obs = torch.tensor(obs).float()

        with torch.no_grad():
            prediction = self.Q.forward(obs)
        print("PREDICITON", prediction)
        prediction = prediction.numpy()

        coin = np.random.uniform()
        
        # take greedy action
        if coin <= 1 - self.epsilon:
            action = np.argmax(prediction)
            
        #else select a random action
        else:
            action = np.random.uniform(low = 0, high = len(prediction) - 1)


        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
