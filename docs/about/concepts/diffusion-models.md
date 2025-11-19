---
description: "How diffusion models work for video generation in NeMo DFM, including EDM and Flow Matching paradigms"
categories: ["concepts-architecture"]
tags: ["diffusion", "video-generation", "edm", "flow-matching"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "explanation"
---

(about-concepts-diffusion-models)=

# Diffusion Models for Video

Diffusion models generate video by learning to reverse a gradual noise-addition process. NeMo DFM implements two paradigms—EDM and Flow Matching—each offering distinct training dynamics and sampling characteristics for video generation.

## Core Mechanism

Diffusion models operate through two complementary processes:

1. **Forward (noise addition)**: The model gradually corrupts clean video data by adding Gaussian noise over many timesteps until the data becomes indistinguishable from pure noise. This forward process is deterministic and follows a predefined noise schedule that controls the rate of corruption.

2. **Reverse (denoising)**: The model learns to invert the forward process by predicting and removing noise at each timestep. During training, the model sees corrupted data at various noise levels and learns to estimate the original clean data or the noise that was added. During inference, the model starts with random noise and iteratively denoises it to generate new video content.

The key insight is that learning to denoise at all noise levels enables generation: if you can remove noise step by step, you can transform random noise into coherent video.

### Video-Specific Challenges

Video diffusion extends image diffusion with additional complexity:

- **Temporal consistency**: Models must maintain coherent motion and object identity across frames. This typically requires 3D attention mechanisms that attend across both spatial and temporal dimensions, or causal attention that processes frames sequentially.
- **Computational scale**: A 5-second video at 24 fps contains 120 frames. Generating each frame at 512×512 resolution requires processing over 31 million pixels, making efficient architectures and parallelization essential.
- **Conditioning mechanisms**: Text embeddings from encoders such as T5 provide semantic guidance, but video generation often requires additional conditioning on motion, camera movement, or reference frames.
- **Memory requirements**: Processing multiple frames simultaneously demands substantial GPU memory. Latent diffusion models compress videos into lower-dimensional representations before applying diffusion, reducing memory usage by 16-64×.

## Diffusion Paradigms in DFM

NeMo DFM implements two paradigms with different mathematical formulations and sampling characteristics:

### EDM (Elucidating Diffusion Models)

EDM frames diffusion as a Stochastic Differential Equation (SDE) where the forward process adds noise according to a continuous-time stochastic process, and the reverse process learns to integrate backward through time.

**Mathematical formulation**: EDM uses a variance-preserving SDE formulation where the noise schedule is parameterized to maintain consistent signal-to-noise ratios across timesteps. The model predicts either the noise ε, the denoised data x₀, or the score function ∇log p(x).

**Sampling characteristics**:

- Stochastic sampling paths allow controlled randomness during generation
- Classifier-free guidance scales the conditional and unconditional predictions: `output = unconditional + guidance_scale × (conditional - unconditional)`
- Typical inference requires 25-50 sampling steps, with quality improving at higher step counts
- Second-order samplers (Heun, DPM-Solver++) can reduce required steps

**When to use EDM**:

- Production inference where generation quality is critical
- Scenarios requiring classifier-free guidance for prompt adherence
- Models trained with variance-preserving objectives

**Primary architecture**: DiT (Diffusion Transformer)

### Flow Matching

Flow matching learns a deterministic ordinary differential equation (ODE) that transports samples from a noise distribution to the data distribution through continuous-time flows.

**Mathematical formulation**: Instead of learning to denoise at discrete timesteps, flow matching learns a velocity field v(x, t) that defines how samples should move through space over time. The generative process integrates this ODE: dx/dt = v(x, t). The training objective directly matches the learned velocity field to a target conditional flow.

**Sampling characteristics**:

- Deterministic sampling paths provide consistent generation given the same seed
- Typically requires fewer sampling steps (10-20) compared to EDM due to the direct ODE formulation
- Time-shift techniques can adjust the speed of the flow at different timesteps
- ODE solvers (Euler, Runge-Kutta) control the numerical integration accuracy

**When to use Flow Matching**:

- Applications requiring deterministic generation for reproducibility
- Scenarios where faster inference (fewer steps) is prioritized
- Research exploring flow-based generative models
- Models trained with flow matching objectives

**Primary architecture**: WAN

## Training Dynamics

### EDM Training Objective

EDM training optimizes the model to predict noise at randomly sampled timesteps. For each training sample, the framework corrupts the clean video by adding Gaussian noise at a random noise level t, then trains the model to estimate either the added noise ε, the clean data x₀, or the score ∇log p(x_t). The loss function typically uses mean squared error between the prediction and target:

`L = E[||prediction - target||²]`

The random sampling of timesteps ensures the model learns to denoise at all noise levels, from slight corruptions to nearly pure noise. Variance-preserving formulations maintain signal strength across timesteps, preventing the model from focusing disproportionately on certain noise levels.

### Flow Matching Training Objective

Flow matching training optimizes the model to predict velocity fields that transport noise to data. The framework samples a clean video, constructs a conditional flow path from noise to that specific video, then trains the model to predict the velocity field along that path:

`L = E[||v_θ(x_t, t) - u_t(x_t)||²]`

where v_θ is the learned velocity field and u_t is the target conditional velocity. The key difference from EDM is that flow matching learns a direct mapping through time rather than iterative denoising. Conditional flow matching uses simple linear interpolation paths during training, making the training objective straightforward while still enabling complex generation.

## Inference Characteristics

### EDM Sampling

EDM sampling iteratively denoises random noise by reversing the learned diffusion process. Starting from pure Gaussian noise, the sampler makes multiple predictions at decreasing noise levels, each time removing a portion of the noise. The sampling trajectory can be deterministic or stochastic depending on the sampler choice.

Classifier-free guidance modifies the sampling process by computing both conditional (text-guided) and unconditional predictions at each step, then extrapolating away from the unconditional prediction. Higher guidance scales (typically 7-15 for video) increase prompt adherence but can reduce diversity. The guidance computation doubles the inference cost since the model must make two predictions per step.

Sampling quality depends on the number of steps and sampler algorithm. First-order samplers (DDPM, DDIM) require more steps but are simpler, while second-order samplers (Heun, DPM-Solver++) achieve similar quality with 50-70% fewer steps by using higher-order numerical approximations.

### Flow Matching Sampling

Flow matching sampling integrates the learned velocity field forward through time using an ODE solver. Starting from noise, the solver numerically integrates dx/dt = v(x, t) from t=0 to t=1, where the velocity field guides the sample along a continuous path toward the data distribution.

The deterministic nature of ODE integration means the same seed and hyperparameters produce identical outputs, which benefits reproducibility and iterative refinement. Time-shift techniques can reweight the integration schedule to spend more computational budget at critical phases of generation.

Flow matching typically achieves competitive quality with fewer function evaluations (10-20) compared to EDM because the direct velocity prediction avoids the iterative error accumulation of denoising steps. However, classifier-free guidance is less commonly used with flow matching, as the formulation doesn't naturally separate conditional and unconditional paths.

## Text Conditioning Mechanisms

Both paradigms condition generation on text prompts through embedding-based guidance:

**Text encoder integration**: Models typically use T5 or CLIP text encoders to convert prompts into high-dimensional embeddings (for example, 768 or 1024 dimensions). These embeddings are injected into the diffusion model through cross-attention layers, where the model's hidden states attend to the text representations at each layer of the architecture.

**Classifier-free guidance**: During training, the model randomly drops conditioning information (typically 10-20% of samples) to learn both conditional p(x|text) and unconditional p(x) distributions. During inference, the two predictions are combined: `output = unconditional + guidance_scale × (conditional - unconditional)`. This extrapolation increases the influence of the text condition, improving prompt adherence at the cost of reduced diversity.

**Negative prompts**: Some implementations support negative text conditioning, which guides generation away from undesired content by subtracting the influence of negative prompt embeddings from the positive prompt guidance. The modified guidance becomes: `output = unconditional + guidance_scale × (positive_conditional - negative_conditional)`.

## Architecture Implementations

### DiT (Diffusion Transformer)

DiT applies transformer architectures to diffusion models by treating the latent video representation as a sequence of patches. Each frame is divided into spatial patches (similar to Vision Transformers), and the patches are processed through transformer blocks with both spatial and temporal attention.

**Key architectural components**:

- **Patch embedding**: Divides frames into non-overlapping patches and projects them to the model dimension
- **Positional encoding**: Combines spatial (2D position within frame) and temporal (frame index) positional information
- **Attention patterns**: 3D attention across height, width, and time dimensions enables modeling spatial structure and temporal dynamics simultaneously
- **Adaptive layer normalization (AdaLN)**: Conditions the normalization on timestep and text embeddings, modulating the network behavior based on the current noise level and prompt
- **Hierarchical processing**: Some variants use multi-scale representations with downsampling and upsampling stages

DiT architectures scale effectively with model size and training compute, making them suitable for large-scale video generation.

### WAN (Flow-Based Architecture)

WAN implements flow matching with architectural designs optimized for learning velocity fields. While sharing transformer-based components with DiT, WAN modifications support the continuous-time dynamics of flow matching.

**Flow-specific design choices**:

- Velocity prediction heads that output per-patch velocity fields
- Time embeddings that integrate smoothly across the continuous [0,1] interval rather than discrete diffusion timesteps
- Architectural modifications that support deterministic ODE integration during inference

The WAN architecture demonstrates that flow matching can achieve competitive results with specialized architectural considerations for the flow-based training paradigm.

## Hyperparameters and Trade-offs

### Noise Schedule

The noise schedule defines the variance of noise at each timestep, controlling the diffusion process trajectory. Common schedules include:

**Linear schedule**: Noise variance increases linearly from near-zero to one. Simple but can be suboptimal for complex data distributions.

**Cosine schedule**: Uses a cosine function to allocate more capacity to mid-range noise levels where the model learns the most semantic information. Generally produces better results than linear schedules.

**Learned schedules**: Some advanced formulations learn the optimal noise schedule during training, adapting to the specific data distribution.

During inference, the schedule determines the timesteps at which the model makes predictions. Non-uniform schedules can concentrate sampling steps at critical noise levels, improving efficiency.

### Guidance Scale

The guidance scale parameter γ controls the strength of conditional guidance in the formula: `output = unconditional + γ × (conditional - unconditional)`.

**Trade-offs**:

- γ = 1: No guidance, equivalent to standard conditional generation
- γ = 7-10: Typical range for video, balances prompt adherence and quality
- γ = 15+: Strong guidance, may improve text alignment but can reduce diversity and introduce artifacts
- γ < 1: Weakens conditioning, increases diversity

Higher guidance scales amplify the difference between conditional and unconditional predictions, effectively increasing the model's confidence in prompt-related features.

### Inference Steps

The number of function evaluations during sampling determines the quality-speed trade-off:

**EDM typical ranges**:

- 25-50 steps: Standard quality, 2-5 seconds per video (depending on resolution and hardware)
- 50-100 steps: High quality, diminishing returns above 50
- <25 steps: Fast sampling, potential quality degradation with first-order samplers

**Flow matching typical ranges**:

- 10-20 steps: Competitive quality due to direct velocity prediction
- 20-50 steps: Marginal improvements, higher computational cost

Second-order ODE solvers can reduce required steps by 30-50% while maintaining quality through better numerical approximation of the integration path.

