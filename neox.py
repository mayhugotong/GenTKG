import torch
import os
import torch.nn as nn
from minimal20b import create_model, create_tokenizer
import glob

class BreakOuterLoop(Exception):
    pass

def init_neox(path_model, # like .../GPTNeoX/20B_checkpoints/global_step150000
              ):
    model_meta_directory = os.path.dirname(path_model)
    path_tokenizer = glob.glob(model_meta_directory+'/*tokenizer.json')[0] # like .../GPTNeoX/20B_checkpoints/20B_tokenizer.json
    model = create_model(path_model, use_cache=True,device='cuda:0',)
    tokenizer = create_tokenizer(path_tokenizer)
    return model, tokenizer
                            

def greedy_generate(m, k, model: nn.Module, input_ids: torch.Tensor, max_seq_len: int,
                    verbose=True):
    """Generate greedily from 20B.

    :param model: NeoX20BModel
    :param input_ids: token IDs [batch_size, seq_len]
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    initial_input_length = input_ids.shape[1]
    current_input_ids = input_ids
    max_seq_len = initial_input_length + max_seq_len  # It is enough to output only 30 more tokens
    layer_past = None
    layer_past_length = 0
    all_token_ids = input_ids.tolist()
    batch_size = len(all_token_ids)
    
    trange = range(initial_input_length, max_seq_len)

    input_length = current_input_ids.shape[1]
    model_out, layer_past = model(
        current_input_ids,
        layer_past=layer_past,
    )

    top_10_indices = torch.topk(model_out[:, -1], k=k, dim=-1).indices
    greedy_predicted_token_ids = top_10_indices[:, m]  # 
    current_input_ids = greedy_predicted_token_ids[:, None]
    l = []
    l.append(greedy_predicted_token_ids.item())

    try:
        should_break = False  # Initialize flag variable to False
        for _ in trange:  # Specify the iteration range appropriately
            input_length = current_input_ids.shape[1]
            model_out, layer_past = model(
                current_input_ids,
                layer_past=layer_past,
            )

            greedy_predicted_token_ids = model_out[:, -1].argmax(-1)

            current_input_ids = greedy_predicted_token_ids[:, None]
            layer_past_length += input_length

            for i in range(batch_size):
# l.append(greedy_predicted_token_ids[i]) #Written before if, \n will also be recorded.
                 if greedy_predicted_token_ids[i].item() == 187:# When a newline is encountered, there is no need to continue predicting later.
                     should_break = True #Set the flag variable to True, indicating that you need to break out of the loop
                     raise BreakOuterLoop
                 l.append(greedy_predicted_token_ids[i])

            if should_break:
                 break # Check the flag variable in the outer loop and break out of the loop

    except BreakOuterLoop:
        pass

    return l


def text_generation(m, k, model: nn.Module,
                         tokenizer,
                         initial_str: str,
                         max_seq_len: int,
                         device=torch.device("cuda:0"), # 0. 
                         verbose=True):
    """Generate greedily from 20B.

    :param model: NeoX20BModel
    :param tokenizer: NeoX20B tokenizer
    :param initial_str: initial string to start generation from
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param device: device to use
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    tokenized = tokenizer.encode(initial_str)
    if len(tokenized.ids) > 1919: # Calculate this value separately for each data set
# input_ids = torch.LongTensor([tokenized.ids[0:90] + tokenized.ids[-1919:]]).to(device)
        input_ids = torch.LongTensor([tokenized.ids[-2009:]]).to(device) # nip mode
        # The input is too long and exceeds the max_length of the model. The truncation operation is performed.
    else:
        input_ids = torch.LongTensor([tokenized.ids]).to(device)

    try:
        all_token_ids = greedy_generate(m, k, model=model, input_ids=input_ids, max_seq_len=max_seq_len, verbose=verbose)
    except BreakOuterLoop:
        pass

    decoded_str = tokenizer.decode(all_token_ids)

    # The following part is to obtain output that conforms to the format. The special symbol # is used to facilitate statistics.
    # Output control is more mature in llama2 and llama2-FT
    if len(decoded_str)< 2:
        return '"#'+str(m)+'"' # Output when exception occurs #0...#9
    elif decoded_str[1].isdigit():
    # If the second predicted word is a number, it is normal
        return decoded_str
    else:
        return '"#'+str(m)+'"' # Output when exception occurs #0...#9