# OllamaVisionKit for ComfyUI

A powerful node suite for advanced image interrogation and automated batch tagging using local Ollama vision models. This toolkit transforms your ComyUI workflow, turning hours of manual data preparation into a streamlined, one-click process.

---

## 📋 Prerequisites

Before you begin, please ensure you have the following set up:

1.  **Ollama Installed and Running**: This plugin requires a local installation of [Ollama](https://ollama.com/). Make sure the Ollama application is running in the background before you start ComfyUI.
2.  **A Vision Model Pulled**: You need at least one vision language model (VLM) installed. We highly recommend starting with `qwen2.5vl:7b` for its excellent performance and balance. You can pull it by running this command in your terminal:
    ```bash
    ollama pull qwen2.5vl:7b
    ```
## ✨ Key Features

*   **🧠 Advanced Prompt Engineering**: Go beyond simple interrogation. Use the `Advanced Configurator` to give your vision models precise instructions, including a "Zero-Tolerance Mode" to prevent AI hallucination, and full control over output style and detail level.
*   **🏭 High-Performance Batch Processing**: Process entire datasets with ease. The `Batch Tagger` node features multi-threaded concurrency to dramatically speed up your workflow, complete with a real-time progress bar in the UI.
*   **🎛️ Granular Content Control**: Use the `Content Options` node to specify exactly what the AI should focus on. Need to describe only the character and ignore the background for LoRA training? Just toggle the switches.
*   **⚙️ Dynamic & User-Friendly**: The nodes automatically detect your installed Ollama vision models and present them in a convenient dropdown menu. No more manual typing!
*   **🔗 Dual-Mode Operation**: Includes both a `Single Interrogator` for quick tests and creative exploration, and a `Batch Tagger` for heavy-duty, industrial-scale data preparation.

## 🚀 Installation

Once you have met the prerequisites:

1.  Clone this repository into your `ComfyUI/custom_nodes/` directory:
    ```bash
    git clone https://github.com/YourUsername/OllamaVisionKit.git
    ```
    
2.  Alternatively, you can download the `__init__.py` file and place it inside a new folder named `OllamaVisionKit` within `ComfyUI/custom_nodes/`.

3.  Restart ComfyUI. You should now find the new nodes under the `Ollama` category.


## 📖 Node Reference

### 🧠 Ollama Advanced Configurator
This node generates the detailed instruction prompt that will be sent to the Ollama model.

*   **`zero_tolerance_mode`**: (Boolean) If `True`, adds a strict instruction for the model to avoid making up details that are not clearly visible. Highly recommended for accurate LoRA training data.
*   **`output_style`**: (Dropdown)
    *   `Natural Language Paragraph`: Outputs a fluid, descriptive paragraph. Great for understanding an image or for blog posts.
    *   `Comma-Separated Tags`: Outputs a list of ke