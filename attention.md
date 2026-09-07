# 1. compare every query with every key
# 2. scale the resultent
# 3. apply the casual mask on them 
# 4. then convert them into probabilities, means converting into attention weights
# 5. retrieve the values/information from V


## 1.
["i", "love", "you", "champ"]

Q = [
    [  # batch_size = 1    
        [0.2, -0.5, 0.8], -> token 1's vector 
        [0.1,  0.4, 0.7], -> token 2's vector
        [-0.3, 0.9, 0.2], -> token 3's vector
        [0.6, -0.1, 0.5]  -> token 4's vector
    ]
]

- Q = [batch_size, sen_len, d_k]
- Q = [batch_size, row, column]
- Q = [1, 4, 3]

- K, Q, V = [batch_size, sen_len, d_k]

## 2.
- when scaling (scores/sqrt), we will take the sqrt of column(d_k)
- because we are taking the dot product of vector, not the batch, not the tokens

## 3.
- applying masking to unwanted/future tokens by making them -ve infinity and then softmax turn them into 0 probobility

## 4.
- applying softmax on the scaled matrix and convert them into probabilities, means converting into attention weights

## 5.
- retrieve the values/information from V and doing the dot product with attention weights 
- attention weights determine how much information from each value vector contributes to the new representation of each query token