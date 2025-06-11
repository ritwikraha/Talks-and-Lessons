---
title: "Can we really scale RL?"
datePublished: Sun Jun 08 2025 10:50:19 GMT+0000 (Coordinated Universal Time)
cuid: cmbnjkroi000102jy6gjl8h86
slug: can-we-really-scale-rl
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1749379209306/457dbf1e-1f2c-4495-a6eb-478208e2b516.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1749379776455/22dd9c14-4149-4b1d-bf85-da08bb900bca.png
tags: machine-learning, reinforcement-learning, llm, gpt, large-language-models, rlhf, claudeai

---

Yes and No. LLM reasoning research is just a big pile of math. We stir the math every once in a while, and it starts doing crazy stuff. For months the community has argued that RL post-training just *polishes* ideas an LLM already had. ProRL politely says:

> “Give me 2k RL steps and a truck-load of tasks and I’ll **invent** **new** reasoning strategies.”

The authors(from Nvidia) back that up with the strongest 1.5 B reasoning model to date, `Nemotron-Research-Reasoning-Qwen-1.5B`, beating its own 7B parameter big brother on several benchmarks.

![ProRL is a recent paper from Nvidia that shows benefits of scaling RL for prolonged duration](https://cdn.hashnode.com/res/hashnode/image/upload/v1749378437797/f9375771-ad89-414a-b937-63218fc7000f.png align="center")

## TL;DR: Prolonged RL is Prolonged RL!

ProRL is designed to extend RL training over a long duration, incorporating several innovative techniques to ensure stability and exploration. The methodology includes:

* **KL Divergence Control**: This technique maintains policy entropy to prevent drift, using a penalty term in the loss function:
    

$$L_{KL-RL}(\theta) = L_{GRPO}(\theta) - \beta D_{KL}(\pi_\theta || \pi_{ref}).$$

* This ensures the policy doesn’t deviate too far from the reference, preserving natural language coherence.
    
* **Reference Policy Resetting**: Periodic resets are applied to avoid premature convergence, ensuring the model continues to explore new reasoning strategies throughout training. This is crucial for long-horizon RL, where models might otherwise get stuck in local optima.
    

**Diverse Task Suite**: ProRL leverages a comprehensive dataset of 136K problems across domains such as math, code, STEM, logical puzzles, and instruction following. The training dataset details are outlined in Table 1 below, showcasing the variety and quantity of tasks:

| **Data Type** | **Reward Type** | **Quantity** | **Data Source** |
| --- | --- | --- | --- |
| Math | Binary | `40k` | DeepScaleR Dataset |
| Code | Continuous | `24k` | Eurus-2-RL Dataset |
| STEM | Binary | `25k` | SCP-116K Dataset |
| Logical Puzzles | Continuous | `37k` | Reasoning Gym |
| Instruction Following | Continuous | `10k` | Llama-Nemotron |

Does it help? Apparently the results speak for themselves

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1749378543339/85d29895-23f4-48e2-a968-5b4aa25fb944.png align="center")

## What is RL again?

The fundamental RL objective in language models is to maximise the expected reward while maintaining proximity to a reference policy. Mathematically, this is expressed as:

$$J(\theta) = \mathbb{E}{\tau \sim \pi\theta}[R(\tau)] - \beta \cdot D_{KL}[\pi_\theta || \pi_{ref}]$$

Where:

* $$\pi_\theta \text{ is the current policy (language model) with parameters } \theta$$
    
* $$\tau \text{ represents a trajectory (prompt + generated response)}$$
    
* $$R(\tau) \text{ is the reward function}$$
    
* $$\pi_{ref} \text{ is the reference policy (typically the initial supervised fine-tuned model)}$$
    
* $$\beta \text{ is the KL penalty coefficient}$$
    
* $${KL} \text{ is the Kullback-Leibler divergence}$$
    

Great now that we have recapped, what is RL, let us focus on the algorithm the Nvidia folks use.

### Group Relative Policy Optimisation (GRPO)

ProRL uses GRPO as its core algorithm. As does every other paper in 2025. Thanks Chinese hedge fund team. The GRPO objective is:

$$L_{GRPO}(\theta) = \mathbb{E}{\tau \sim \pi\theta}\left[\min\left(r_\theta(\tau)A(\tau), \text{clip}(r_\theta(\tau), 1-\epsilon, 1+\epsilon)A(\tau)\right)\right]$$

Where the probability ratio is:

$$r_\theta(\tau) = \frac{\pi_\theta(\tau)}{\pi_{old}(\tau)}$$

`Imagine the model is solving "What is 15 × 7?"`

* Old policy might generate `"15 × 7 = 105"` with probability `0.3`
    
* New policy generates the same response with probability `0.45`
    
* Then `rθ = 0.45/0.3 = 1.5`
    

This ratio tells us the new policy is `1.5×` more likely to generate this response than the old policy.

### The Advantage Function A(τ)

$$A(\tau) = \frac{R_\tau - \text{mean}({R_i}{i \in G(\tau)})}{\text{std}({R_i}{i \in G(\tau)})}$$

Here, `G(`τ`)` represents a group of trajectories sampled together, typically 16 in this implementation.

‘This is GRPO's key innovation. Instead of comparing against a learned value function, it compares against other responses in the same batch.

`Let us assume there are 4 responses to the same math problem:`

* Response 1: Correct answer, clear steps → `R₁ = 1.0`
    
* Response 2: Correct answer, messy work → `R₂ = 0.8`
    
* Response 3: Wrong answer → `R₃ = 0.0`
    
* Response 4: Partially correct → `R₄ = 0.4`
    

`Mean reward = (1.0 + 0.8 + 0.0 + 0.4) / 4 = 0.55` `Standard deviation ≈ 0.41`

* For Response 1: `A(τ₁) = (1.0 - 0.55) / 0.41 ≈ 1.10` (positive advantage) ✅
    
* For Response 3: `A(τ₃) = (0.0 - 0.55) / 0.41 ≈ -1.34` (negative advantage)❌
    

### The Clipping Mechanism

ProRL also uses asymmetric clipping:

$$\text{clip}(r_\theta(\tau), 1-\epsilon_{low}, 1+\epsilon_{high})$$

With `ϵlow=0.2` and `ϵhigh=0.4`

* If advantage is positive: ratio clipped to \[0.8, 1.4\]
    
* If advantage is negative: ratio clipped to \[0.6, 1.2\]
    

This asymmetry encourages exploration. We will discuss more on this later. In a nutshell, when the model finds a good response (positive advantage), it can increase its probability by up to `40%`. But when penalizing bad responses, it's more consertvative.

## Scaling is hard in RL

Yeah, you cant just throw truckloads of compute and expect it to work. Why? A couple of reasons:

1. ### Entropy Collapse
    

Entropy measures how "spread out" the model's predictions are.

A critical challenge in prolonged RL training is entropy collapse. The entropy of a policy is defined as:

$$H(\pi_\theta) = -\mathbb{E}{a \sim \pi\theta}[\log \pi_\theta(a|s)]$$

As training progresses, the policy tends to become increasingly deterministic, leading to:

$$\lim_{t \to \infty} H(\pi_\theta^t) \to 0$$

`Early in training, for the prompt "What is 2+2?", the model might output`:

* `"4"` with probability `0.3`
    
* `"The answer is 4"` with probability `0.25`
    
* `"2+2 equals 4"` with probability `0.25`
    
* Other variations with probability `0.2`
    

`High entropy ≈ 1.95` (lots of diversity)

After extensive training without entropy control:

* `"The answer is 4"` with probability `0.95`
    
* Everything else with probability `0.05`
    

`Low entropy ≈ 0.20` (almost deterministic)

This collapse means the model stops exploring new ways to solve problems, getting stuck in local optima. This collapse severely limits exploration and learning of new strategies.

2. ### KL Divergence Growth
    

Without constraints, the KL divergence between the current and reference policies grows unbounded:

$$D_{KL}(\pi_\theta || \pi_{ref}) = \mathbb{E}{s \sim \rho^{\pi\theta}}\left[\sum_a \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}\right]$$

This drift can lead to:

1. Loss of linguistic coherence
    
2. Reward hacking
    
3. Catastrophic forgetting of pre-trained capabilities
    

## Gimme, solutions man!

So yeah, we have established why scaling RL is not that easy. So what do we do? Or what have the authors done?

1. ### Modified Loss Function with KL Regularization
    

ProRL modifies the GRPO loss to include explicit KL regularization:

$$L_{KL-RL}(\theta) = L_{GRPO}(\theta) - \beta D_{KL}(\pi_\theta || \pi_{ref})$$

The KL divergence term expands to:

$$D_{KL}(\pi_\theta || \pi_{ref}) = \mathbb{E}{s \sim \rho^{\pi\theta}}\left[\sum_a \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}\right]$$

`For the prompt "Explain gravity", suppose at a particular generation step`:

| **Tokens** | **Reference model's next token probabilities:** | **Current model's probabilities**: |
| --- | --- | --- |
| `“Gravity”` | `0.4` | `0.1` (much lower!) |
| `“The”` | `0.3` | `0.7` (much higher!) |
| `"Newton's"` | `0.2` | `0.15` |
| `“Others”` | `0.1` | `0.05` |

The KL contribution from just these tokens:

$$D_{KL} = 0.1 \log(0.1/0.4) + 0.7 \log(0.7/0.3) + 0.15 \log(0.15/0.2) + ...$$

$$≈0.1(−1.39)+0.7(0.85)+0.15(−0.29)+...≈0.41$$

With `β = 0.1` this adds a penalty of `0.041` to the loss, discouraging the model from drifting too far from sensible language patterns. This maintains a balance between reward optimisation and staying close to the reference distribution.

2. ### Decoupled Clipping (DAPO Integration)
    

ProRL incorporates asymmetric clipping bounds.

$$\text{clip}(r_\theta(\tau), 1-\epsilon_{low}, 1+\epsilon_{high})$$

With `ϵlow=0.2` and `ϵhigh=0.4`, this encourages exploration by allowing larger upward movements in probability space for previously unlikely actions.

We discussed this previously but lets look at a more illustrative example now:

`"Find the derivative of f(x) = x³ + 2x² - 5x + 3"`

One way to think of DAPO's asymmetric clipping is like teaching a student:

* When they discover a brilliant solution → Celebrate enthusiastically! (larger positive updates) 🥳
    
* When they make mistakes → Correct gently (conservative negative updates) 🤔
    

This asymmetry prevents the model from becoming overly cautious while still maintaining stability.

Okay enough analogies, let us take a look at the example at hand:

`"Find the derivative of f(x) = x³ + 2x² - 5x + 3"`

### `Scenario 1: Excellent Response (Positive Advantage)`✅

| Aspect | Details |
| --- | --- |
| **Model's Response** | `"To find f'(x), I'll differentiate term by term: f'(x) = 3x² + 4x - 5"` |
| **Quality Assessment** | ✅ Correct answer ✅ Clear step-by-step explanation ✅ Proper mathematical notation |
| **Reward** | `1.0` (perfect score) |
| **Group Context** | Other responses in batch averaged 0.4 reward |
| **Advantage Score** | A(τ) = +1.46 (this response is much better than average) |

| Probability Update Process | Standard PPO | DAPO |
| --- | --- | --- |
| **Current probability (π\_old)** | 15% | 15% |
| **Desired probability (π\_new)** | 30% (model wants to double it) | 30% (same desire) |
| **Probability ratio (r)** | 30% ÷ 15% = 2.0 | 30% ÷ 15% = 2.0 |
| **Clipping bound** | \[0.8, 1.2\] | \[0.8, 1.4\] |
| **Actual ratio after clipping** | 1.2 (capped) | 1.4 (more room) |
| **Final probability** | 15% × 1.2 = **18%** | 15% × 1.4 = **21%** |
| **Improvement allowed** | +20% max | +40% max |

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">DAPO allows 2× more aggressive learning for good solutions (40% vs 20% increase)</div>
</div>

### `Scenario 2: Incorrect Response (Negative Advantage)`

| Aspect | Details |
| --- | --- |
| **Model's Response** | `"The derivative is x² + 2x - 5"` |
| **Quality Assessment** | ❌ Wrong answer (missed coefficient 3) ❌ Incomplete differentiation ✅ At least attempted the problem |
| **Reward** | `0.0` (incorrect) |
| **Group Context** | Other responses averaged `0.4` reward |
| **Advantage Score** | `A(τ) = -0.97` (this response is worse than average) |

| Probability Update Process | Standard PPO | DAPO |
| --- | --- | --- |
| **Current probability (π\_old)** | 20% | 20% |
| **Desired probability (π\_new)** | 5% (model wants to reduce it significantly) | 5% (same desire) |
| **Probability ratio (r)** | 5% ÷ 20% = 0.25 | 5% ÷ 20% = 0.25 |
| **Clipping bound** | \[0.8, 1.2\] | \[0.8, 1.4\] |
| **Actual ratio after clipping** | 0.8 (capped) | 0.8 (same cap for negative) |
| **Final probability** | 20% × 0.8 = **16%** | 20% × 0.8 = **16%** |
| **Reduction allowed** | \-20% max | \-20% max (conservative) |

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">DAPO remains equally conservative for bad solutions to prevent catastrophic forgetting</div>
</div>

3. ### Reference Policy Resetting
    

Periodically, ProRL resets the reference policy:

$$\pi_{ref}^{(k+1)} \leftarrow \pi_\theta^{(k)}$$

`Before Reset (Run 3)`**:**

* Current model has learned good math strategies
    
* KL divergence has grown to 0.15 (getting large)
    
* Model wants to explore new approaches but is held back
    

`After Reset (Run 4)`**:**

* Reference model now includes all learned improvements
    
* KL divergence resets to 0.
    
* Model can freely explore from this new baseline
    

It's like a rock climber establishing a new base camp at a higher altitude before continuing the ascent.

This prevents the KL term from dominating the loss and allows continued improvement. The reset points are strategically chosen based on validation performance.

4. ## Training Dynamics and Scaling Laws
    

### Performance Scaling

The paper demonstrates that both pass@1 and pass@k scale with training steps. The relationship can be approximated as:

$$\text{Pass@k}(t) = 1 - (1 - p_0)e^{-\alpha t}$$

Where:

* $$t \text{ is the training step}$$
    
* $$p_0 \text{ is the initial performance}$$
    
* $$\alpha \text{ is the learning rate coefficient}$$
    

**Example with actual numbers:**

* Initial pass@1 for AIME problems: p0=0.285(28.5%)
    
* Learning rate coefficient: α ≈ 0.0003
    
* After 1000 steps: Pass@1 ≈ 1 - (1-0.285)e^{-0.3} ≈ 0.481 (48.1%)
    
* After 2000 steps: Pass@1 ≈ 1 - (1-0.285)e^{-0.6} ≈ 0.602 (60.2%)
    

This shows sustained improvement rather than quick saturation.

### 4.2 Reasoning Boundary Expansion

The paper introduces the concept of reasoning boundary, measured by pass@k metrics. For a task `x`, the upper bound is:

$$\mathbb{E}{x,y \sim D}[\text{pass@k}] \leq 1 - \frac{(1 - \mathbb{E}{x,y \sim D}[\rho_x])^2 + \text{Var}(\rho_x)}{k/2}$$

Where:

$$\rho_x \text{ is the pass@1 accuracy for task }x$$

5. ## Got any code?
    

Let us look at a *minimal* sketch of the ProRL loop—GRPO loss, KL penalty, and periodic reference resets.

<div data-node-type="callout">
<div data-node-type="callout-emoji">⚠</div>
<div data-node-type="callout-text">Warning: Boilerplate code. Wouldn’t run! Try to understand not blindly copy paste!</div>
</div>

Replace the stubbed reward with your verifier of choice.

```python
import torch, copy, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributions import Categorical

MODEL_ID = "Qwen/Qwen-1.5B" # <-- yes its qwen, deal with it
EPS_LOW, EPS_HIGH = 0.2, 0.4
KL_BETA = 0.1

model = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda()
ref_model = copy.deepcopy(model).cuda()
tok = AutoTokenizer.from_pretrained(MODEL_ID)

opt = torch.optim.AdamW(model.parameters(), lr=2e-6)

def grpo_loss(logits, old_logits, adv):
    dist, old = Categorical(logits=logits), Categorical(logits=old_logits)
    ratio = torch.exp(dist.logits - old.logits)
    clipped = torch.clamp(ratio, 1-EPS_LOW, 1+EPS_HIGH)
    return -(torch.min(ratio*adv, clipped*adv).mean())

def kl_penalty(logits, ref_logits):  # symmetric KL is fine here
    p, q = Categorical(logits=logits), Categorical(logits=ref_logits)
    return KL_BETA * torch.distributions.kl_divergence(p, q).mean()

def rollout(prompt, temp=1.2):
    """Sample a single response & a mock scalar reward."""
    input_ids = tok(prompt, return_tensors="pt").to('cuda')
    out = model.generate(**input_ids, do_sample=True, temperature=temp,
                         max_new_tokens=128, return_dict_in_generate=True,
                         output_scores=True)
    # take logits for sampled tokens
    logits = torch.stack(out.scores, dim=1).squeeze(0)
    reward = torch.tensor([random.random()], device='cuda')  # <-- Pls replace reward here
    return logits, reward

prompts = ["Integrate x^2 * sin(x).", "Write a BFS in brainfuck.", "Color this 12-node graph."]

for step in range(2000):            # pretend-marathon
    batch_logits, batch_old, rewards = [], [], []
    for p in prompts:
        logits, r = rollout(p)
        batch_logits.append(logits); rewards.append(r)
        batch_old.append(logits.detach())               # store old logits

    adv = torch.cat(rewards) - torch.mean(torch.cat(rewards))  # simple baseline
    logits = torch.cat(batch_logits)
    old_logits = torch.cat(batch_old)

    loss = grpo_loss(logits, old_logits, adv) + \
           kl_penalty(logits, ref_model(**tok(prompts, return_tensors="pt").to('cuda')).logits)

    opt.zero_grad(); loss.backward(); opt.step()

    # hard reset every 250 or so steps
    if (step + 1) % 250 == 0:
        ref_model = copy.deepcopy(model).cuda()
        print(f"Reset ref @ step {step+1}, loss {loss.item():.4f}")
```

*What does this mean?*

1. **GRPO** uses only log-probabilities—no value network headaches.
    
2. **KL penalty** tethers the policy to a moving reference.
    
3. Every 250 steps we “cut the rope” and start a new KL leash, mimicking the paper’s resets.
    

## So, can we really scale RL?

ProRL’s key insight is refreshingly old-school: **keep training longer AND keep it stable.** With the right guard-rails (KL + resets) and a buffet of varied tasks, even a tiny 1.5B model can punch far above its weight, inventing answer trajectories its base version and frankly other dinosaur models (looking at you Llama) had never dreamed of.

That’s both inspiring and a sobering reminder that *compute is still king*. Happy experimenting :)

Key Citations

* [ProRL Paper Abstract](https://arxiv.org/abs/2505.24864)
    
* [ProRL on Hugging Face](https://huggingface.co/papers/2505.24864)
    
* [RLHF Blog](https://huggingface.co/blog/rlhf)
    
* [PPO Paper](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
    
* [Hugging Face Model Weights](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)