from file_path import File_path
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import torch
import os
import threading


class BallModel():
    """Model loader for the GBRAG pipeline.

    Handles three model roles:
      - sentence embedding model (SentenceTransformer)
      - generator LLM (Ollama endpoint or local Transformers weights)
      - propositionizer (T5-style seq2seq model for proposition extraction)
    """

    def __init__(self, file_path: str):
        """file_path: absolute path of the current source file."""
        self.file_path = file_path
        self.file_path_tool = File_path(self.file_path)

    def _get_embedding_model(self):
        # Fall back to CPU when no GPU is available (local CPU debugging);
        # override with GBRAG_EMBEDDING_DEVICE.
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(model_name_or_path=str(self.file_path_tool._get_embedding_model_path()),
                                   device=os.environ.get("GBRAG_EMBEDDING_DEVICE", default_device))

    def _get_llm_model(self):
        """Load the generator LLM.

        Two backends are supported, selected by GBRAG_LLM_BACKEND:
          - "ollama"      (default): a local Ollama server; point
                          GBRAG_LLM_BASE_URL to its port (default http://localhost:8080).
                          The timeout makes a stuck Ollama fail fast instead of
                          blocking forever (override with GBRAG_LLM_TIMEOUT).
          - "transformers": local Hugging Face weights loaded with Transformers;
                          set GBRAG_LLM_MODEL_PATH to the model directory.
        """
        backend = os.environ.get("GBRAG_LLM_BACKEND", "ollama")
        if backend == "transformers":
            return self._get_llm_model_transformers()
        return ChatOllama(
            # Model served by the Ollama endpoint; override with GBRAG_LLM_MODEL.
            model=os.environ.get("GBRAG_LLM_MODEL", "qwen2.5:14b"),
            temperature=0.0,
            base_url=os.environ.get("GBRAG_LLM_BASE_URL", "http://localhost:8080"),
            timeout=float(os.environ.get("GBRAG_LLM_TIMEOUT", "300")),
        )

    def _get_llm_model_transformers(self):
        """Local Transformers causal LM backend (no network, no Ollama)."""
        # Default to a repo-local directory; override with GBRAG_LLM_MODEL_PATH.
        default_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "llm",
        )
        model_path = os.environ.get("GBRAG_LLM_MODEL_PATH", default_model_dir)

        if not os.path.isfile(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(
                f"No config.json found in the model directory: {model_path}. "
                "Set GBRAG_LLM_MODEL_PATH to a downloaded Hugging Face model."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )

        tokenizer.padding_side = "left"

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # When GPU memory is tight, force a lower precision with GBRAG_LLM_DTYPE:
        # auto (default) / float16 / bfloat16 / float32. For further savings set
        # GBRAG_LLM_8BIT=1 to load with bitsandbytes 8-bit quantization.
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        llm_dtype = dtype_map.get(
            os.environ.get("GBRAG_LLM_DTYPE", "auto"), "auto"
        )
        llm_8bit = os.environ.get("GBRAG_LLM_8BIT", "0") == "1"

        # GBRAG_LLM_DEVICE_MAP controls which GPUs hold the model:
        # auto (default) / cuda:0 / cuda:1 / balanced, etc.
        # On a multi-GPU QA run, pinning one GPU (e.g. cuda:1) leaves the other
        # for the T5 propositionizer and the embedding model.
        # With spread placement, use GBRAG_LLM_MAX_MEM to cap per-GPU memory
        # (GiB, comma separated, e.g. 16,16) instead of relying on the detected
        # free memory, which can over-commit a single card.
        llm_device_map = os.environ.get("GBRAG_LLM_DEVICE_MAP", "auto")
        max_memory = None
        max_mem_env = os.environ.get("GBRAG_LLM_MAX_MEM", "")
        if max_mem_env:
            max_memory = {
                idx: f"{gb.strip()}GiB"
                for idx, gb in enumerate(max_mem_env.split(","))
                if gb.strip()
            }
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=llm_dtype,
            device_map=llm_device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            load_in_8bit=llm_8bit,
        )

        llm_model.eval()

        # Serialize concurrent generate() calls to avoid GPU memory spikes.
        _generate_lock = threading.Lock()

        # Ablation runs can enable sampling with GBRAG_LLM_SAMPLE=1;
        # the default 0 means greedy decoding.
        llm_sample = os.environ.get("GBRAG_LLM_SAMPLE", "0") == "1"

        def invoke_transformers(prompt_value):
            # Accept both LangChain ChatPromptTemplate and PromptTemplate inputs.
            if hasattr(prompt_value, "to_messages"):
                langchain_messages = prompt_value.to_messages()

                role_map = {
                    "system": "system",
                    "human": "user",
                    "ai": "assistant",
                }

                messages = [
                    {
                        "role": role_map.get(message.type, "user"),
                        "content": str(message.content),
                    }
                    for message in langchain_messages
                ]

            elif hasattr(prompt_value, "to_string"):
                messages = [
                    {
                        "role": "user",
                        "content": prompt_value.to_string(),
                    }
                ]

            else:
                messages = [
                    {
                        "role": "user",
                        "content": str(prompt_value),
                    }
                ]

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=28672,
            )

            # device_map="auto" may spread the model across one or more GPUs.
            input_device = next(llm_model.parameters()).device

            model_inputs = {
                key: value.to(input_device)
                for key, value in model_inputs.items()
            }

            with _generate_lock, torch.inference_mode():
                # With GBRAG_LLM_SAMPLE=1 use sampling (do_sample=True);
                # otherwise stay greedy (equivalent to temperature=0, used by
                # all full-pipeline experiments).
                generated_ids = llm_model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=llm_sample,
                    temperature=(0.7 if llm_sample else 1.0),
                    top_p=(0.9 if llm_sample else 1.0),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            # Drop the prompt tokens; decode only the newly generated tokens.
            generated_ids = generated_ids[
                            :, model_inputs["input_ids"].shape[1]:
                            ]

            response_text = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            # The rest of the project reads response.content, so return an AIMessage.
            return AIMessage(content=response_text)

        return RunnableLambda(invoke_transformers)

    def _get_pro_model(self):
        """Load the propositionizer (T5-style seq2seq) model.

        The propositionizer must run on a GPU; without CUDA this raises an
        error instead of silently falling back to CPU, which would slow the
        graph-building stage by orders of magnitude.
        For deliberate CPU debugging, set GBRAG_ALLOW_CPU=1 explicitly.
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            if os.environ.get("GBRAG_ALLOW_CPU", "0") != "1":
                raise RuntimeError(
                    "No CUDA GPU detected: the propositionizer must run on a GPU. "
                    "Check free cards with nvidia-smi and select one via "
                    "GBRAG_CUDA_DEVICES; for deliberate CPU debugging, set "
                    "GBRAG_ALLOW_CPU=1."
                )
            device = "cpu"
        model_name_or_path = self.file_path_tool._get_pro_model_path()
        is_local_path = os.path.exists(model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=is_local_path,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            local_files_only=is_local_path,
        ).to(device)

        if device == "cuda":
            torch.cuda._lazy_init()

        return {
            "tokenizer": tokenizer,
            "model": model,
            "device": device
        }
