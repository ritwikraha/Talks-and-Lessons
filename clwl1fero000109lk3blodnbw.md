---
title: "Understanding PaliGemma in 50 minutes  or less"
seoTitle: "Understanding PaliGemma"
seoDescription: "Understand PaliGemma: a versatile vision-language model for image captioning, visual question answering, and object detection in under 50 minutes"
datePublished: Fri May 24 2024 18:51:09 GMT+0000 (Coordinated Universal Time)
cuid: clwl1fero000109lk3blodnbw
slug: understanding-paligemma-in-50-minutes-or-less
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1716576227391/18576908-1fb3-4719-ba00-2d1a9e1ce917.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1716576492483/c58007d0-6ee6-4ff9-a9b6-039976af5699.png
tags: ai, machine-learning, google, computer-vision, deep-learning, huggingface, transformers, gemini, gemma, gemma-ai

---

PaliGemma is designed as a versatile model for transfer to a wide range of vision-language tasks such as image and short video caption, visual question answering, text reading, object detection, and object segmentation.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1716574313229/6b4d62db-951a-448e-95bc-9fa134ef39bd.gif align="center")

<details data-node-type="hn-details-summary"><summary>Note of some importance</summary><div data-type="detailsContent">A note of thanks, acknowledgment, and warning: This post is a collation of wonderful resources from the <a target="_blank" rel="noopener noreferrer nofollow" href="https://developers.googleblog.com/id/gemma-family-and-toolkit-expansion-io-2024/" style="pointer-events: none">PaliGemma</a> model card, the <a target="_blank" rel="noopener noreferrer nofollow" href="https://huggingface.co/blog/paligemma" style="pointer-events: none">HuggingFace Blogpost</a>, and the <a target="_blank" rel="noopener noreferrer nofollow" href="https://github.com/google-research/big_vision" style="pointer-events: none">BigVision</a> repository. The point of this post is to simplify and show it from my perspective. I am open to constructive criticism and</div></details>

## What is PaliGemma?

PaliGemma is a new family of vision-language models from Google. These models can process both images and text to produce text outputs.

Google has released three types of PaliGemma models:

* **Pretrained (pt) models**: Trained on large datasets without task-specific tuning.
    
* **Mix models**: A combination of pre-trained and fine-tuned elements.
    
* **Fine-tuned (ft) models**: Optimized for specific tasks with additional training.
    

Each type comes in different resolutions and multiple precisions for convenience. All models are available on the Hugging Face Hub with model cards, licenses, and integration with transformers.

## "How do I get it running?"

The model comes with simple out-of-the-box usage with Huggingface Transformers. A simple colab notebook is linked [here](https://github.com/ritwikraha/Talks-and-Lessons/blob/main/notebooks/paligemma_inference.ipynb).

```python

from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from huggingface_hub import notebook_login
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import requests

input_text = "Who is this person?"
img_url = "https://huggingface.co/datasets/ritwikraha/random-storage/resolve/main/cohen.jpeg"
input_image = Image.open(requests.get(img_url, stream=True).raw)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = PaliGemmaProcessor.from_pretrained(model_id)

inputs = processor(text=input_text, images=input_image,
                  padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
model.to(device)
inputs = inputs.to(dtype=model.dtype)

with torch.no_grad():
  output = model.generate(**inputs, max_length=496)
print(processor.decode(output[0], skip_special_tokens=True)
)
```

## **"What is the architecture like?"**

PaliGemma (GitHub) is a family of vision-language models with an architecture featuring SigLIP-So400m as the image encoder and Gemma-2B as the text decoder.

* **SigLIP** is a state-of-the-art model capable of understanding both images and text. Similar to CLIP, it includes an image and text encoder trained together.
    
* **Gemma** is a decoder-only model designed for text generation.
    

### But what is SigLIP?

SigLIP introduces a straightforward modification to the widely-used CLIP architecture, as detailed in the paper [https://arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343). CLIP's architecture includes an image encoder and a text encoder, both utilizing Transformer-based models.

The pre-training of CLIP employs a contrastive approach to ensure that embeddings of corresponding images and texts are close in the embedding space, while non-matching pairs are positioned far apart. Traditionally, CLIP is trained with a softmax loss function, which necessitates a global view of all pairwise similarities for probability normalization.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1716661503131/850b71f2-0c3b-4564-81d2-ad61a67443c1.png align="center")

SigLIP simplifies this process by substituting the softmax loss with a sigmoid loss. Unlike softmax, the sigmoid loss does not require a global perspective on pairwise similarities. This modification converts the task into a binary classification problem: determining whether a given image and text pair belong together, with a straightforward yes or no.

### But what is Gemma?

Gemma is a family of lightweight, state-of-the-art open models derived from the research and technology that underpinned the creation of Gemini models. These models exhibit robust performance across various academic benchmarks, particularly in language understanding, reasoning, and safety. The Gemma models are available in two sizes, featuring 2 billion and 7 billion parameters, with both pretrained and fine-tuned checkpoints provided.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1716661686222/0517ac7b-fa76-4f4a-b163-e198ca2d8e11.png align="center")

PaliGemma combines these two components: the image encoder from SigLIP and the text decoder from Gemma, connected through a linear adapter. This combination creates a powerful vision-language model that can be pre-trained on image-text data and fine-tuned for tasks like captioning and referring segmentation.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1716573006637/22bef36d-1b58-4413-b122-bbd9022f06ad.png align="center")

Let us look at the boilerplate code of what this might look like. The actual code is present in the `bigvision` [repository](https://github.com/google-research/big_vision/tree/main/big_vision/trainers/proj/paligemma) and this is simply a reduced version of that code.

```python
class Model(nn.Module):
    """Two towers transformer for image and text."""
    img_model: str = "vit"
    img: Optional[ConfigDict] = None
    llm_model: str = "proj.paligemma.gemma_bv"
    llm: Optional[ConfigDict] = None

    def setup(self):
        # Initialize LLM and image models
        self._llm = importlib.import_module(f"big_vision.models.{self.llm_model}").Model(**(self.llm or {}), name="llm")
        img_config = {"num_classes": self._llm.embdim, **(self.img or {})}
        self._img_model = importlib.import_module(f"big_vision.models.{self.img_model}").Model(**img_config, name="img")

    def embed_image(self, image, train=False):
        """Embeds the input image."""
        # Preprocess image and call image model
        # Return image embeddings and any auxiliary outputs
        pass

    def embed_text(self, tokens, train=False):
        """Embeds the input text tokens."""
        # Call LLM to embed text tokens
        # Return text embeddings and any auxiliary outputs
        pass

    def embed_image_and_text(self, image, text, input_mask=None, mask_ar=None, train=False):
        """Concatenates image and text embeddings."""
        # Embed image and text separately
        # Combine embeddings into a single sequence
        pass

    def __call__(self, image, text, mask_ar, train=False):
        """Processes input image and text and returns logits."""
        # Embed image and text
        # Create attention mask and call transformer
        # Extract and return logits
        pass

    def prefill_cache(self, x, input_mask, mask_ar, cache_size):
        """Initializes decoding cache with prompt."""
        # Initialize cache for decoding
        pass

    def extend_cache(self, x):
        """Advances decoding cache with new input."""
        # Extend cache for decoding
        pass

    def _fallback_prefill_cache(self, x, input_mask, mask_ar, cache_size):
        # Fallback method for initializing cache
        pass

    def _fallback_extend_cache(self, x):
        # Fallback method for extending cache
        pass
```

The above boilerplate code defines a `Model` class in a PyTorch-like framework, designed to implement a two-tower transformer architecture for vision-language models (VLMs). The model consists of two primary components: an image model (using ViT) and a language model (using a variant of the Gemma model).

The `Model` class includes methods for embedding images and text, concatenating these embeddings, and processing them together to generate output logits.

### Two-Tower Transformer Architecture for VLMs

The two-tower transformer architecture in vision-language models (VLMs) involves separate towers (or networks) for processing images and text, which are later combined for joint tasks. Here's how it works:

1. **Image Embedding Tower**:
    
    * The image model, specified by `img_model` (e.g., ViT), processes input images.
        
    * The model is initialized with parameters specified in the `img` configuration.
        
    * The `embed_image` method preprocesses the image and generates embeddings using the image model.
        
2. **Text Embedding Tower**:
    
    * The language model, specified by `llm_model` (e.g., Gemma variant), processes input text tokens.
        
    * The model is initialized with parameters specified in the `llm` configuration.
        
    * The `embed_text` method generates embeddings for text tokens using the language model.
        
3. **Combining Embeddings**:
    
    * The `embed_image_and_text` method separately embeds images and text, then concatenates these embeddings into a single sequence.
        
    * This combined sequence is used for tasks that require joint image-text understanding.
        
4. **Processing Inputs**:
    
    * The `__call__` method processes input images and text, creates an attention mask, and passes the combined embeddings through a transformer to generate output logits.
        

By leveraging this two-tower approach, the PaliGemma model can effectively learn and utilize the relationships between visual and textual information, which is not wholly indifferent from how a Vision language Model is pre-trained.

As mentioned before the PaliGemma release includes three types of models:

* **PT checkpoints** are pre-trained models that can be further fine-tuned for specific downstream tasks.
    
* **Mix checkpoints**: These models are pre-trained and then fine-tuned on a mixture of tasks. They are suitable for general-purpose inference with free-text prompts and are intended for research purposes only.
    
* **FT checkpoints**: These are specialized fine-tuned models, each optimized for a different academic benchmark. They come in various resolutions and are also intended for research purposes only.
    

<details data-node-type="hn-details-summary"><summary>PaliGemma Model Sizes</summary><div data-type="detailsContent">The PaliGemma models come in three different resolutions: <code>224x224</code>, <code>448x448</code>, and <code>896x896</code>. They are also available in three different precisions: <code>bfloat16</code>, <code>float16</code>, and <code>float32</code>. Each model repository contains checkpoints for a specific resolution and task, with three revisions corresponding to the available precisions. The main branch of each repository contains <code>float32</code> checkpoints, while the <code>bfloat16</code> and <code>float16</code> revisions contain the respective precision models.</div></details>

There are separate repositories for models compatible with HuggingFace Transformers and those using the original JAX implementation.

## What was the Pretraining like?

PaliGemma is pre-trained on the following mixture of datasets:

### Datasets

* **WebLI (Web Language Image)**:
    
    ![The WebLI dataset. Top: Sampled images 4 associated with multilingual... |  Download Scientific Diagram](https://www.researchgate.net/publication/363564241/figure/fig2/AS:11431281084502540@1663211858376/The-WebLI-dataset-Top-Sampled-images-4-associated-with-multilingual-alt-text.png align="center")
    
    A web-scale multilingual image-text dataset sourced from the public web. Various splits of WebLI are used to develop versatile model capabilities such as visual semantic understanding, object localization, visually situated text understanding, and multilingual proficiency.
    
* **CC3M-35L**:
    
    ![a) Three image-text pairs randomly sampled from CC3M dataset have some... |  Download Scientific Diagram](https://www.researchgate.net/publication/369655531/figure/fig1/AS:11431281132722392@1680233052185/a-Three-image-text-pairs-randomly-sampled-from-CC3M-dataset-have-some-local.png align="left")
    
    Curated English image-alt\_text pairs from webpages (Sharma et al., 2018). Translated into 34 additional languages using the Google Cloud Translation API.
    
* **VQ²A-CC3M-35L/VQG-CC3M-35L**:  
    A subset of VQ2A-CC3M (Changpinyo et al., 2022a), translated into the same 34 languages as CC3M-35L, using the Google Cloud Translation API.
    
* **OpenImages**:
    
    ![Open Images V7 - Description](https://storage.googleapis.com/openimages/web/images/oidv7_all-in-one_example_ab.jpg align="left")
    
    Detection and object-aware questions and answers (Piergiovanni et al. 2022) generated by handcrafted rules on the OpenImages dataset.
    
* **WIT**:  
    Images and texts collected from Wikipedia (Srinivasan et al., 2021).
    

## "Wait can I finetune it?"

### Fine-Tuning Methods

**1\. JAX Fine-Tuning Script:**

* PaliGemma was trained in the `big_vision` codebase, which has also been used for models like BiT, ViT, LiT, CapPa, and SigLIP.
    
* The project configuration folder `configs/proj/paligemma/` contains a [`README.md`](http://README.md).
    
* Pretrained models can be transferred using configuration files in the `transfers/` subfolder.
    
* To transfer your own model, fork `transfers/`[`forkme.py`](http://forkme.py) and follow the instructions in the comments to adapt it to your use case.
    
* A Colab notebook, `finetune_paligemma.ipynb`, provides a simplified fine-tuning process on a free T4 GPU runtime, updating only the weights in the attention layers (170M parameters) and using SGD instead of Adam.
    

**2\. Fine-Tuning with Hugging Face Transformers:**

* Fine-tuning PaliGemma is straightforward using the `transformers` library.
    
* Methods such as QLoRA or LoRA fine-tuning can be employed.
    
* An example process involves briefly fine-tuning the decoder, followed by switching to QLoRA fine-tuning.
    
* Ensure to install the latest version of the `transformers` library.
    

**3\. Fine-Tuning with Vanilla Pytorch script**

* A small and lean PyTorch script to fine-tune the PaliGemma model
    
* Developed by @[Aritra Roy Gosthipaty](@ariG23498) in this [repository](https://github.com/ariG23498/ft-pali-gemma).
    
* Fine-tune on any dataset containing images and caption pairs.
    

### Training and Model Information

* PaliGemma models have been released in various fine-tuned versions by Google.
    
* These models were trained using the `big_vision` codebase, with a history of developing models like BiT, ViT, LiT, CapPa, SigLIP, and more.
    

### Model Performance Table on Fine-tuned Checkpoints

| Model Name | Dataset/Task | Score in Transferred Task |
| --- | --- | --- |
| paligemma-3b-ft-vqav2-448 | Diagram Understanding | 85.64 Accuracy on VQAV2 |
| paligemma-3b-ft-cococap-448 | COCO Captions | 144.6 CIDEr |
| paligemma-3b-ft-science-qa-448 | Science Question Answering | 95.93 Accuracy on ScienceQA Img subset with no CoT |
| paligemma-3b-ft-refcoco-seg-896 | Understanding References to Specific Objects in Images | 76.94 Mean IoU on refcoco |
| paligemma-3b-ft-rsvqa-hr-224 | Remote Sensing Visual Question Answering | 92.61 Accuracy on test |

## Why PaliGemma?

Changing the pretraining strategy and utilizing larger datasets like LAION can significantly enhance PaliGemma's capabilities as a multimodal model for various tasks. Pretraining on vast and diverse datasets improves the model's understanding and generation of nuanced and contextually rich outputs. By scaling the architecture, such as replacing the autoregressive decoder with a more advanced model like Gemini, and training the SigLIP processor on higher-quality, finer-grained images, PaliGemma can achieve superior performance in tasks requiring detailed visual-semantic understanding, precise object localization, and robust multilingual text generation. This will eventually lead to the model becoming more versatile and powerful for a wide range of multimodal applications.