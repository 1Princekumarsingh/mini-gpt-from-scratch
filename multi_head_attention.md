## nn.linear(features_in, features_out)

- let's say: 
d_model = 4
d_ff = 6

- PyTorch creates:
weight.shape = [6, 4]
bias.shape   = [6]

So conceptually:
W =
[
 [w11 w12 w13 w14]   ← output neuron 1
 [w21 w22 w23 w24]   ← output neuron 2
 [w31 w32 w33 w34]
 [w41 w42 w43 w44]
 [w51 w52 w53 w54]
 [w61 w62 w63 w64]
]

b = [b1,b2,b3,b4,b5,b6]

# view()
- changes how we interpret the same underlying data, when the memory layout permits it.
- work only when memory is contiguous.

- after transpose(), tensor memory may no longer be contiguous, that why we use it.

## self.q_proj, self.k_proj, self.v_proj
- perform linear transformation in which informations are mixed (can assume 1 subpart of vector can store the info almost of whole vector)
- we slice it based on the no. of head

## out_proj
- basically it mix the informations from multiple head 