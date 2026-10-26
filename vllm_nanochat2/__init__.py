"""Plugin do vLLM para modelos nanochat convertidos (arquitectura Nanochat2ForCausalLM)."""


def register():
    from vllm import ModelRegistry

    if "Nanochat2ForCausalLM" not in ModelRegistry.get_supported_archs():
        # importação diferida, para não inicializar CUDA no processo principal
        ModelRegistry.register_model("Nanochat2ForCausalLM", "vllm_nanochat2.model:Nanochat2ForCausalLM")
