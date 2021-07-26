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

# Summary for article "Deconvolution and Checkerboard Artifacts"

- Observations: uneven overlap happens when kernel size not divisible by the stride
- Stride 1 deconvolution removes artifacts that divide their size and reduces artifacts that less than their size
- Not only GAN's problem
- GAN: not generator, discriminator and gradient
- Solution:
    - Stride is divided by kernel size
    - nearest neighbor resize or bilinear interpolation then convolution

# Wasserstein GANs with Gradient Penalty 
- Mode: peak of the distribution
- Real dataset: multiple modes 
- Mode Collapse: Happens in multimodal settings and get stuck in one mode
- Problems of BCE loss: vanishing gradient problem (When 2 distributions are far apart). Discriminator is easier to train than Generator
- Earth Mover's Distance:
    - Effort to move the generated distribution to real distribution (distance + amount)
    - Unlike BCE loss which approaches 0 or 1, gradient of EMD is linear
- W-Loss:
    - $min_g max_c E(c(x)) - E(c(g(z)))$ (Output real number)
    - Critic: Maximize the distance between fake example and real example
- 1-L Continuous: The norm of the gradient should be at most 1 for every point (stability, not growth too much)
    - Weight clipping: force weight of the critic in the fixed interval
    - GD to update weights $\rightarrow$ clip the critic's weights 
    - Limitation: Limit the ability to learn of the critic
- Gradient Penalty: Adding the regularization term 
    $\lambda E(||\nabla c(\hat x)||_2 - 1)^2$


# Conditional GAN and Controllable Generation