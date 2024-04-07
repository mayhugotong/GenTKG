#Add an additional parameter m to the function, and loop the generate function once for each predicted word.
import torch
import torch.nn as nn
from tqdm import auto as tqdm_lib



class BreakOuterLoop(Exception):
    pass

def greedy_generate(m, model: nn.Module, input_ids: torch.Tensor, max_seq_len: int,
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

    top_10_indices = torch.topk(model_out[:, -1], k=10, dim=-1).indices
    greedy_predicted_token_ids = top_10_indices[:, m]  #
    current_input_ids = greedy_predicted_token_ids[:, None]
    l = []
    l.append(greedy_predicted_token_ids.item())

    try:
        should_break = False #Initialize the flag variable to False
        for _ in trange: # Specify the iteration range appropriately
            input_length = current_input_ids.shape[1]
            model_out, layer_past = model(
                current_input_ids,
                layer_past=layer_past,
            )

            greedy_predicted_token_ids = model_out[:, -1].argmax(-1)

            current_input_ids = greedy_predicted_token_ids[:, None]
            layer_past_length += input_length

            for i in range(batch_size):
# l.append(greedy_predicted_token_ids[i]) #Written before if, \n will be recorded
                if greedy_predicted_token_ids[i].item() == 187:# When a carriage return is encountered, there is no need to continue predicting later.
                    should_break = True #Set the flag variable to True to indicate the need to break out of the loop
                    raise BreakOuterLoop
# if greedy_predicted_token_ids[i].item() == 209:
# should_break = True # Set the flag variable to True to indicate the need to break out of the loop
# raise BreakOuterLoop
                l.append(greedy_predicted_token_ids[i])

            if should_break:
                break # Check the flag variable in the outer loop and break out of the loop

    except BreakOuterLoop:
        pass

    return l


def greedy_generate_text(m, model: nn.Module,
                         tokenizer,
                         initial_str: str,
                         max_seq_len: int,
                         device=torch.device("cuda:0"), # default 0. 
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
    if len(tokenized.ids) > 2020: # For icews14 it is 2020; for icews18 it is 2009; for ecola it is 2029
        input_ids = torch.LongTensor([tokenized.ids[-2020:]]).to(device)
        #The input is too long and exceeds the max_length of the model.
    else:
        input_ids = torch.LongTensor([tokenized.ids]).to(device)

    try:
        all_token_ids = greedy_generate(m, model=model, input_ids=input_ids, max_seq_len=max_seq_len, verbose=verbose)
    except BreakOuterLoop:
        pass
# all_token_ids = greedy_generate(m, model=model, input_ids=input_ids, max_seq_len=max_seq_len, verbose=verbose)

# print('Generated initial text: ', tokenizer.decode(all_token_ids)[0:2] )
    decoded_str = tokenizer.decode(all_token_ids)
    if len(decoded_str)< 2:
        return '"#'+str(m)+'"' # Output when exception occurs #0...#9
    elif decoded_str[1].isdigit():
    # If the second predicted word is a number, it is normal
        return tokenizer.decode(all_token_ids)
    else:
        return '"#'+str(m)+'"' # Output when exception occurs #0...#9











