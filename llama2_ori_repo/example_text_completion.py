# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        # "I believe the meaning of life is",
        # "Simply put, the theory of relativity states that ",
        """1620: [iran, make an appeal or request, 70.israel] 
1635: [iran, make an appeal or request, 97.hassan] 
1695: [iran, make an appeal or request, 2.president] 
1770: [iran, make an appeal or request, 97.hassan] 
2100: [iran, make an appeal or request, 97.hassan] 
2220: [iran, make an appeal or request, 97.hassan] 
2265: [iran, make an appeal or request, 813.molotov] 
2310: [iran, make an appeal or request, 102.the us] 
2385: [iran, make an appeal or request, 98.tehran] 
2400: [iran, make an appeal or request, 2.president] 
2430: [iran, make an appeal or request, 66.regime] 
2445: [iran, make an appeal or request, 70.israel] 
2670: [iran, make an appeal or request, 137.united nations] 
2685: [iran, make an appeal or request, 95.protester] 
2685: [iran, make an appeal or request, 54.canada] 
2715: [iran, make an appeal or request, 311.pakistan] 
2730: [iran, make an appeal or request, 27.united states] 
2730: [iran, make an appeal or request, 54.canada] 
2775: [iran, make an appeal or request, 27.united states] 
2805: [iran, make an appeal or request, 1192.federal judiciary] 
2895: [iran, make an appeal or request, 70.israel] 
2910: [iran, make an appeal or request, 144.north korea] 
2925: [iran, make an appeal or request, 66.regime] 
2925: [iran, make an appeal or request, 95.protester] 
2940: [iran, make an appeal or request, 949.iraq] 
2940: [iran, make an appeal or request, 27.united states] 
2940: [iran, make an appeal or request, 66.regime] 
2940: [iran, make an appeal or request, 632.terrorist organization] 
2940: [iran, make an appeal or request, 2962.al haq] 
2985: [iran, make an appeal or request, 27.united states] 
3015: [iran, make an appeal or request, 5.government] 
3075: [iran, make an appeal or request, 68.iranian] 
3105: [iran, make an appeal or request, 68.iranian] 
3165: [iran, make an appeal or request, 70.israel] 
3165: [iran, make an appeal or request, 19.ruler] 
3210: [iran, make an appeal or request, 54.canada] 
3225: [iran, make an appeal or request, 5.government] 
3510: [iran, make an appeal or request, 1394.voice of america] 
3615: [iran, make an appeal or request, 544.turkish] 
3705: [iran, make an appeal or request, 102.the us] 
3840: [iran, make an appeal or request, 27.united states] 
3885: [iran, make an appeal or request, 1096.dissident] 
3945: [iran, make an appeal or request, 54.canada] 
3945: [iran, make an appeal or request, 2.president] 
4155: [iran, make an appeal or request, 275.new york] 
4155: [iran, make an appeal or request, 70.israel] 
4185: [iran, make an appeal or request, 21.media] 
4275: [iran, make an appeal or request, 343.google] 
4320: [iran, make an appeal or request, 97.hassan] 
4320: [iran, make an appeal or request, 343.google] 
4560: [iran, make an appeal or request,""",
        # Few shot prompt (providing a few examples before asking model to complete more);
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
    ]
    results = generator.text_completion(0,
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
