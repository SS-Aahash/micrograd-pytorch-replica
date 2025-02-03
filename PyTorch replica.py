import math
import random
# this is a just a value(not a neuron or node) but with a bunch of functionality, something which we store inside the neur on/node. 
class value:
    # "children" will contain children nodes which formed this parent node, TYPE:- tuple
    # "op" is the operation which was performed between the children to form the current node
    def __init__(self,data,children=(),op=''):
        self.data = data
        self.prev = set(children)
        self.op = op
        self.grad = 0
        self.backward = lambda: None
    
    def __repr__(self):# for representation purpose ,when ever print an object of this class we'll be returned this, if it weren't for thid we'll be getting some shaddy address of that object  
        return f"value(data={self.data})"
    
    def __sub__(self,other):
        return self + (-1*other)
    
    def __add__(self,other):
        other = other if isinstance(other,value) else value(other) #if we're tryin to add value object with int then this converts that int into an value object
        out =  value(self.data + other.data,(self,other),"+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward
        return out
    
    def __mul__(self,other):
        other = other if isinstance(other,value) else value(other)
        out =  value(self.data * other.data,(self,other),"*")
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = backward
        return out

    def __radd__(self,other):# if we're tryin to add int with the value object then this function is called , this fuc just reverses the order(value obj + int) and return the ans 
        return self + other
    
    def __rmul__(self,other): #same as abv but with multiplication
        return self*other

    def __rsub__(self,other):
        return self-other

    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = value(t,(self,))
        def backward():
            self.grad += (1-t**2) * out.grad # diffrentiation of tan(x) = 1-( tan(x) )^2
        out.backward = backward
        return out

    def exp(self): # computes e^x
        x = self.data
        out = value(math.exp(x),(self,))

        def backward():
            self.grad += out.data*out.grad
        out.backward = backward
        return out
            
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = value(self.data**other,(self,))

        def backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out.backward = backward
        return out
        
    def __truediv__(self,other): # DEV_NOTE:- test for rtruediv
        return self * other**-1

    def auto_backpropogate(self):
        topo = []
        visited = set()
        ''' topological sort reversed =  we arrange all the nodes in a way that front node of the neural net comes first in the sort, children of the first node 
                                         comes after that and so on, coz first we want to calculate the gradient of the first node and based on that calculate the 
                                         gradient of it's children node/neuron'''
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1
        for node in reversed(topo):
            node.backward()

# neuron

#input
# x1 = value(2)
# x2 = value(0)
# #weights
# w1 = value(-3)
# w2 = value(1)
# b = value(6.7)

# x1w1 = x1*w1
# x2w2 = x2*w2
# x1w1_add_x2w2 = x1w1 + x2w2
# n = b + x1w1_add_x2w2
# o = n.tanh()

# manual backpropogation

# o.grad = 1
# o.backward()
# n.backward()
# x1w1_add_x2w2.backward()
# x1w1.backward()
# x2w2.backward()

class neuron:
    def __init__(self,no_of_input):
        self.w = [ value(random.uniform(-1,1)) for _ in range(no_of_input) ]       
        self.b = value(random.uniform(-1,1)) 
    
    def __call__(self,x):
        z = sum( (Wi*Xi for Wi,Xi in zip(self.w,x)) , self.b)
        out = z.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]

class layer:
    def __init__(self,no_of_input,no_of_neuron):
        self.neurons = [neuron(no_of_input) for _ in range(no_of_neuron)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return[p for neuron in self.neurons for p in neuron.parameters()]

class mlp:
    def __init__(self, no_of_input, list_of_nurons_in_each_layer):
        sz = [no_of_input] + list_of_nurons_in_each_layer
        self.layers = [layer(sz[i],sz[i+1]) for i in range(len(list_of_nurons_in_each_layer))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x 
    def parameters(self):
        return[p for layer in self.layers for p in layer.parameters()] 
# making the model
n = mlp(3,[4,4,1]) # this is our Neural network

#dataset
x = [
    [2,3,-1],
    [3,-1,0.5],
    [0.5,1,1],
    [1,1,-1]
]
y = [1,-1,-1,1]



for k in range(100):
    #forward pass
    y_pred = [n(i) for i in x]
    loss = sum((x-y)**2 for x,y in zip(y,y_pred))
    #backward pass
    for p in n.parameters():
        p.grad = 0
    loss.auto_backpropogate()
    #update
    for p in n.parameters():
        p.data -= 0.05 * p.grad # 0.5 is the learning rate of the model
    print(k,loss)

print(y_pred)
