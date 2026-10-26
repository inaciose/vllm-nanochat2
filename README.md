# vllm-nanochat2

Plugin do vLLM para servir modelos nanochat convertidos com `convert_nanochat_to_hf.py`
(arquitectura `Nanochat2ForCausalLM`). Escrito para a API das versões 0.26 e 0.28 do vLLM.
Suporta apenas uma GPU.

## Instalação

No ambiente onde tens o vLLM:

```bash
pip install -e /caminho/para/vllm_nanochat2      # ou: uv pip install -e ...
```

O `pip` regista o plugin no grupo `vllm.general_plugins`, por isso o `vllm serve` e o `LLM(...)` passam a reconhecer a arquitectura automaticamente.

## Testar

Compara a geração greedy do vLLM com a do `transformers` (corre um de cada vez, para caber numa GPU de 12 GB):

```bash
python /caminho/para/vllm_nanochat2/test_vllm_vs_hf.py /files/ws/hf/nanochat-d20-v1-sft
```

## Servir

```bash
vllm serve /files/ws/hf/nanochat-d20-v1-sft \
    --trust-remote-code \
    --served-model-name nanochat-d20 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85
```

`--trust-remote-code` é necessário para ler o `config.json` (a classe de configuração vem no repositório do modelo).
O modelo também pode ser o id do repositório no Hub em vez da pasta local.

Pedido de teste (API compatível com OpenAI):

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nanochat-d20",
  "messages": [{"role": "user", "content": "Olá! Qual é a capital de Portugal?"}],
  "max_tokens": 200
}'
```

Os valores por omissão de geração (temperatura 0,6, top-k 50 e paragem em `<|assistant_end|>`/`<|bos|>`) vêm do `generation_config.json` do modelo.

## Como funciona o smear

No nanochat, cada token recebe uma fracção do embedding do token anterior. No vLLM, o modelo só vê os tokens novos de cada pedido, por isso o plugin tem uma camada de atenção auxiliar, sem pesos:

- Guarda na cache do vLLM o id de cada token, dividido em 4 nibbles (exactos mesmo com cache em fp8).
- Usa uma codificação posicional fixa que faz cada token atender, com peso praticamente 1, só à posição anterior. A margem é de pelo menos 48 no score, numa janela de 513.

Assim, o token anterior vem da própria cache, e tudo funciona com batching, prefill por blocos e cache de prefixos. A camada auxiliar tem a mesma forma e janela das camadas de janela curta, por isso partilha o grupo de cache delas. O vLLM pode avisar que acrescentou camadas de *padding* à cache; o desperdício é pequeno.

## Problemas conhecidos

- Se o arranque falhar durante a captura de CUDA graphs ou a compilação, acrescenta `--enforce-eager`.
- Não suporta tensor/pipeline parallel (um d20 cabe folgadamente numa GPU).
- As chamadas à ferramenta Python (`<|python_start|>` … `<|python_end|>`) aparecem no texto, mas não são executadas.
