---
license: mit
license_link: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE
language:
- en
- fr
pipeline_tag: text-generation
tags:
- nlp
- code
- mlx
inference:
  parameters:
    temperature: 0
widget:
- messages:
  - role: user
    content: Can you provide ways to eat combinations of bananas and dragonfruits?
base_model: microsoft/Phi-3-mini-4k-instruct
library_name: mlx
---
