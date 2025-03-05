# FLAN-T5 Fine-Tuning and Inference

This project explores different approaches for working with **FLAN-T5**, a powerful large language model. It covers **zero-shot and multi-shot inference**, **full fine-tuning**, and **parameter-efficient fine-tuning (PEFT) using LoRA**.

## Why Fine-Tune FLAN-T5?

While FLAN-T5 is a powerful pre-trained model, its **zero-shot performance** may not always generate summaries that are concise, accurate, or aligned with human-written summaries. Fine-tuning helps in:

- **Adapting the model to specific tasks** → Improves performance on **dialogue summarization** by learning patterns from the dataset.  
- **Enhancing coherence and factual consistency** → Fine-tuning reduces hallucinations and makes summaries **more relevant**.  
- **Optimizing for domain-specific data** → Helps the model generate better outputs when trained on relevant dialogue datasets like **DialogSum**.  
- **Comparing full vs. parameter-efficient fine-tuning** → Evaluating **full fine-tuning vs. LoRA** helps determine the best approach for efficiency vs. performance.

Fine-tuning allows us to customize FLAN-T5 for **real-world summarization tasks**, improving accuracy while balancing compute costs.
---

## Project Structure

- **FLAN-T5_Zero_Multishot_Inference.ipynb**  
  - Implements **zero-shot, one-shot, and few-shot inference** using FLAN-T5.  
  - Compares generated summaries with human-written summaries.  

- **FLAN-T5_Full_FineTuning.ipynb**  
  - Fine-tunes FLAN-T5 on the **DialogSum dataset**.  
  - Trains all model parameters, improving summarization quality but requiring high computational resources.  
  - Due to constraints, the model was trained for **one epoch**, but performance can improve with more training.  

- **FLAN-T5_FineTuning_PEFT_LoRA.ipynb**  
  - Uses **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.  
  - Trains only **a small subset of parameters** instead of updating the full model.  
  - Achieves results **close to full fine-tuning** while using significantly less memory and compute.  

---

## Dataset

This project uses the **DialogSum dataset** from Hugging Face:

- **Dataset Name:** `knkarthick/dialogsum`  
- **Description:** A collection of **10,000+ dialogues** with **human-annotated summaries and topics**.  
- **Purpose:** Used for training and evaluating dialogue summarization models.  

---

## How to Use

1. **Run Zero/Multi-Shot Inference**  
   - Open `FLAN-T5_Zero_Multishot_Inference.ipynb` and execute the cells to test the model without fine-tuning.  

2. **Fine-Tune the Model**  
   - If you have enough computational power, run `FLAN-T5_Full_FineTuning.ipynb`.  
   - The script will fine-tune FLAN-T5 on the dataset and save the trained model.  

3. **Use LoRA for Efficient Fine-Tuning**  
   - Run `FLAN-T5_FineTuning_PEFT_LoRA.ipynb` for a **memory-efficient** alternative.  
   - This approach is ideal for limited resources while still improving model performance.  

---

## Results Summary

| Model                 | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|----------------------|---------|---------|---------|------------|
| Zero-Shot (Original) | 0.261   | 0.085   | 0.225   | 0.228      |
| Full Fine-Tuned     | 0.389   | 0.131   | 0.282   | 0.283      |
| LoRA Fine-Tuned     | 0.332   | 0.088   | 0.251   | 0.253      |

- **Full fine-tuning achieved the best results** but required significant compute time.  
- **LoRA fine-tuning performed well** while using **only 1.4% of trainable parameters** compared to full fine-tuning.  

---

## Future Improvements

- Train for **more epochs** to improve fine-tuning results.  
- Experiment with **different LoRA configurations** to balance efficiency and accuracy.
- Deploy the model using **ONNX or TorchScript** for optimized inference.  

---

This project provides a **practical comparison** of different FLAN-T5 fine-tuning approaches, making it easier to choose the right method based on available resources.  
