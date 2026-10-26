#!/usr/bin/env python
"""
Compara a geração greedy do vLLM (com o plugin vllm_nanochat2) com a do transformers,
para o mesmo modelo convertido. Corre primeiro o transformers, liberta a GPU e depois o vLLM.

    python test_vllm_vs_hf.py /files/ws/hf/nanochat-d20-v1-sft

Testa: vários pedidos em simultâneo (batching), um prompt com mais de 512 tokens (janelas
deslizantes), uma conversa com várias mensagens e a repetição dos mesmos pedidos (cache de prefixos).
"""

import argparse
import gc

import torch

PROMPTS = [
    [{"role": "user", "content": "Olá! Qual é a capital de Portugal?"}],
    [{"role": "user", "content": "Explica em duas frases o que é a fotossíntese."}],
    [{"role": "user", "content": "Escreve um haiku sobre o mar de Aveiro."}],
    [
        {"role": "system", "content": "Responde sempre de forma breve."},
        {"role": "user", "content": "Quanto é 7 vezes 8?"},
        {"role": "assistant", "content": "7 vezes 8 são 56."},
        {"role": "user", "content": "E 9 vezes 6?"},
    ],
    [{"role": "user", "content": "Diz-me três rios portugueses."}],
]

LONG_TEXT = (
    "O nanochat é uma implementação compacta de um modelo de linguagem de ponta a ponta: tokenizer, "
    "pré-treino, afinação supervisionada, aprendizagem por reforço, avaliação e interface de conversação. "
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    stop_ids = sorted({tok.convert_tokens_to_ids("<|assistant_end|>"), tok.convert_tokens_to_ids("<|bos|>")})

    prompts = []
    for msgs in PROMPTS:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(tok(text, add_special_tokens=False)["input_ids"])
    long_q = "Resume numa frase o texto seguinte.\n\n" + LONG_TEXT * 14
    text = tok.apply_chat_template([{"role": "user", "content": long_q}], tokenize=False, add_generation_prompt=True)
    prompts.append(tok(text, add_special_tokens=False)["input_ids"])
    print("Comprimento dos prompts:", [len(x) for x in prompts])

    # --- transformers
    print("\n== transformers")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    model.to("cuda").eval()
    hf_out, hf_margin = [], []
    with torch.no_grad():
        for ids in prompts:
            inp = torch.tensor([ids], device="cuda")
            out = model.generate(
                inp,
                attention_mask=torch.ones_like(inp),
                do_sample=False,
                max_new_tokens=args.max_tokens,
                eos_token_id=stop_ids,
                pad_token_id=stop_ids[0],
                return_dict_in_generate=True,
                output_logits=True,  # logits em bruto (sem temperatura/top-k)
            )
            hf_out.append([t for t in out.sequences[0, len(ids):].tolist() if t not in stop_ids])
            # margem entre o 1.º e o 2.º token mais prováveis em cada passo (logits com softcap)
            margins = []
            for step in out.logits:
                top2 = torch.topk(step[0].float(), 2).values
                margins.append((top2[0] - top2[1]).item())
            hf_margin.append(margins)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- vLLM
    print("\n== vLLM")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=stop_ids)
    batch = [{"prompt_token_ids": ids} for ids in prompts]
    runs = [llm.generate(batch, sp), llm.generate(batch, sp)]  # a 2.ª corrida usa a cache de prefixos

    TIE = 0.5  # abaixo disto, a escolha entre os dois melhores tokens depende do ruído de bf16
    print("\n== Resultados")
    all_ok = True
    for r, outs in enumerate(runs, 1):
        print(f"-- corrida {r}{' (cache de prefixos)' if r == 2 else ''}")
        for i, (o, ref) in enumerate(zip(outs, hf_out)):
            got = [t for t in o.outputs[0].token_ids if t not in stop_ids]
            same = 0
            for a, b in zip(got, ref):
                if a != b:
                    break
                same += 1
            full = got == ref
            note = ""
            if not full and same < len(hf_margin[i]):
                m = hf_margin[i][same]
                tie = m < TIE
                all_ok &= tie
                note = f" — margem no ponto de divergência: {m:.3f} ({'quase-empate, normal' if tie else 'SUSPEITO'})"
            print(f"  prompt {i} ({len(prompts[i])} tokens): {same}/{max(len(got), len(ref))} iguais{' ✓' if full else ''}{note}")
            if not full:
                print(f"     hf:   {tok.decode(ref)!r}\n     vllm: {tok.decode(got)!r}")
            elif r == 1:
                print(f"     {tok.decode(got)!r}")
    print(
        "\nOK: o vLLM reproduz o transformers (as divergências que houver são quase-empates de bf16)"
        if all_ok
        else "\nATENÇÃO: há divergências onde o token escolhido pelo HF tinha uma margem clara (ver acima)"
    )


if __name__ == "__main__":
    main()
