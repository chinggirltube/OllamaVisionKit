# OllamaVisionKit

### 【中文版】README.md

# OllamaVisionKit for ComfyUI

一个功能强大的ComfyUI节点套件，专为使用本地Ollama视觉模型进行高级图像解析和自动化批量打标而设计。本工具包将彻底改变您的ComfyUI工作流，将数小时的手动数据准备工作转变为精简、一键式的自动化流程。

---

## 📋 前提条件

在开始之前，请确保您已完成以下设置：

1.  **Ollama 已安装并正在运行**: 本插件需要您在本地安装并运行 [Ollama](https://ollama.com/)。在启动ComfyUI之前，请确保Ollama应用程序正在后台运行。
2.  **已下载视觉模型**: 您至少需要安装一个视觉语言模型（VLM）。我们强烈推荐您从 `qwen2.5vl:7b` 开始，因为它在性能和资源占用之间取得了出色的平衡。您可以在终端（命令行）中运行以下命令来下载它：
    ```bash
    ollama pull qwen2.5vl:7b
    ```

## ✨ 核心功能

*   **🧠 高级指令工程**: 超越简单的图像反推。使用 `Advanced Configurator`（高级配置生成器）节点，为您的视觉模型提供精确的指令，包括防止AI幻觉的“零容忍模式”，以及对输出风格和细节程度的完全控制。
*   **🏭 高性能批量处理**: 轻松处理整个数据集。`Batch Tagger`（批量打标）节点具备多线程并发功能，可极大提升您的工作流速度，并在UI中提供实时进度条。
*   **🎛️ 精细化内容控制**: 使用 `Content Options`（内容选项）节点，精确指定AI应当关注的焦点。在训练LoRA时，需要只描述角色而忽略背景吗？只需拨动开关即可。
*   **⚙️ 动态与用户友好**: 插件会自动检测您已安装的Ollama视觉模型，并以一个方便的下拉菜单呈现。无需再手动输入模型名称！
*   **🔗 双模式操作**: 包含用于快速测试和创意探索的 `Single Interrogator`（单张反推），以及用于重度、工业级数据准备的 `Batch Tagger`（批量打标）。

## 🚀 安装

在满足前提条件后：

1.  将本仓库克隆到您的 `ComfyUI/custom_nodes/` 目录下：
```
git clone https://github.com/chinggirltube/OllamaVisionKit.git
```
    
2.  或者，您也可以下载 `__init__.py` 文件，并将其放入 `ComfyUI/custom_nodes/` 目录下一个名为 `OllamaVisionKit` 的新文件夹中。

3.  重启ComfyUI。您现在应该可以在 `Ollama` 分类下找到新的节点。


## 📖 节点说明

### 🧠 Ollama Advanced Configurator（高级配置生成器）
此节点用于生成将发送给Ollama模型的详细指令。

*   **`zero_tolerance_mode`** (布尔值): 如果为 `True`，将添加一条严格的指令，要求模型避免编造图片中不清晰可见的细节。强烈建议在为LoRA训练准备精确数据时开启。
*   **`output_style`** (下拉菜单):
    *   `Natural Language Paragraph` (自然语言段落): 输出流畅、富有描述性的段落。非常适合用于理解图片或撰写博客文章。
    *   `Comma-Separated Tags` (逗号分隔标签): 输出由关键词和短语组成的列表。是训练数据集的理想选择。
*   **`description_length`** (下拉菜单):
    *   `Concise` (简洁): 要求一个简短的摘要。
    *   `Medium` (中等): 要求一个细节程度均衡的描述。
    *   `Detailed` (详细): 要求一个高度详细的描述，捕捉尽可能多的信息。
*   **输入**: `content_options` (可选) - 在此连接 `Ollama Content Options` 节点。
*   **输出**: `instruction` - 最终生成的、可送入执行器节点的指令。

### 🔧 Ollama Content Options（内容选项）
此节点提供了一份清单，用于控制生成指令中的“分析清单”应包含哪些图像方面。

*   **复选框**: 每个复选框对应一个特定类别（例如，`include_subject` - 包含主体, `include_clothing` - 包含衣物, `include_background` - 包含背景）。勾选或取消这些选项将修改最终的指令。
*   **输出**: `content_options` - 一个包含您所选项的字典，用于连接到 `Advanced Configurator`。
