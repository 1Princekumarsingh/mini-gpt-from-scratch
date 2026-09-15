## ffn
- apply non linear transformation on each token vector.
- Refines/enriches the token representation using the contextual information obtained from attention.

# in residual connection:
we perform the vector addition as them are adding the updates information into the original information/vector.

- why not matrix multiplication? 
1. because their will a chance that the resultant will be s scalar value if the input and attention output is vector. 
2. for matrix multiplication we need: [row, column] [column, row] which is not possible until we perform transpose but the output generated will be a similarity/scored matrix.

- Eg:
x = [x1, x2, x3, x4, x5, x6, x7, x8] and y = [y1, y2, y3, y4, y5, y6, y7, y8]
x⋅y=x1​y1​+x2​y2​+⋯+x8​y8​
[8]@[8]  -> scalar

## matrix/vector multipication and addiction effect
1. x + f
- combine two representations directly
-  output stays [8]

2. x @ W
→ transform/mix the features of x
→ output can stay [8]

3. Q @ Kᵀ
→ compare queries and keys
→ produces attention scores

- matrix/vector multipication: transform/mix info/ similarity table

## torch.nn.LayerNorm(normalized_shape)
(int or list/tuple): Dictates which inner dimensions to normalize over.

## nn.Module is basically a callable object.
- when you instantiate an nn.Module and then call the object with (), PyTorch routes that call to forward(). 
- Other methods are only called when you explicitly call them or when forward() calls them.

                    Creating object:
                    model = MyModel()
                            ↓
                    __init__() runs
                    
                    model(x)
                       ↓
                    __call__()
                       ↓
                    forward(x)

                but:
                    model.some_function(x)
                         ↓
                    some_function(x)

## 
x
 ↓
attention(x)
 ↓
x + attention(x)
 ↓
LayerNorm
 ↓
FFN
 ↓
previous result + FFN(result)
 ↓
LayerNorm
 ↓
output
