# Discriminative vs Generative models

- Discriminative: Distinguish between classes (classifiers)
    - Features $\rightarrow$ class (probability)
    - VAE (Encoder-Decoder)
    - Generative Adversarial Networks
    - P(Y | X)
    - Input: image and output: probability of fake given image
- Generative: Noise, Class $\rightarrow$ features
    - Produce realistic examples
    - P(X | Y)
    - Input: noise 
- Applications: Image Translation, Image production (GauGan)
- Binary Cross Entropy Cost Function
- Problems: Superior discriminator and generator 

# Deep convolution GANS

$$z_i^{[l]} = \sum_{i=0}^n W_i^{[l]} a_i^{[l-1]}$$
$$a_i^{[l]} =  g^{[l]}(z_i^{[l]})$$
- Differentiable: Backpropagation
- Non-linear: compute complex features
- Relu: Dying relu problem, non differentiable at 0
- Sigmoid: Vanish gradient and saturation problems
- Batch normalization:
    - Covariance shift: changes in 1 r.v affects the other r.v (test set is different from the training set distribution)
    - Smooth cost function and reduce training time
    - Test uses the running statistics from training
- Upsampling: Bilinear, linear interpolation
- Transposed Convolutions: center-pixel have been influenced more than other pixel (checkerboard pattern)
    - Note: upsampling = predefined method, transposed convolutions = learnt filter