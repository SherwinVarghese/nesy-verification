Questions:

1. We need the same train split as they use otherwise we will be verifying data that has been used for training - (however using random_split) 
2. Do they standardise the input for training? If so, we need to. If they don't this may be an issue as will harm the bounds.
3. What is the difference between magnitude and parity classification - and why is one the first 3 ouputs and the other the next two?
4. Check LogSoftmax implementation. This may be an issue.


# To give to DKIS

CSV in the shape:
imageId,parity_lb

