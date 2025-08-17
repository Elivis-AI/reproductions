import torch


EOS_TOKEN = '[EOS]'
VOCAB = [EOS_TOKEN, "hello", "world", "how", "are", "you", "doing", "today", "goodbye", "thanks", "please"]

class DummyModel:
    def next_token_proba(self, sequence):
        vocab_size = len(VOCAB)
        # if sequence.ndim == 1:
        return torch.softmax(torch.rand(vocab_size), dim=-1) # return a probability distribution over the vocabulary
        # elif sequence.ndim == 2:
        #     return torch.softmax(torch.rand(len(sequence), vocab_size), dim=-1) # [B, N]
        # else:
        #     raise ValueError(f'Unknow sequence ndim: {sequence.ndim}')

def beam_search(model, start_token, k, max_length):
    """
    Perform beam search to generate the most probable sequence.

    Args:
        model: An object with a method next_token_proba(sequence) -> torch.Tensor.
        start_token (int): The index of the start token in the vocabulary.
        k (int): Beam width, the number of sequences to keep at each step.
        max_length (int): The maximum length of the sequence to generate.

    Returns:
        List[int]: The most probable sequence of token indices.

    Note: 一个正确且高效的算法需要处理的问题大概有两个：
    - 充分利用硬件，可以处理批量数据，且尽量使用并行计算少用循环
    - 处理好长短不同的生成结果
    """
    # Initialize the beam with the start token and a score of 0.0
    sequences = [(0.0, [start_token])]  # Each item is a tuple: (score, sequence)

    epsilon = 1e-10  # Small constant to prevent log(0)

    # Iterate up to the maximum sequence length
    for _ in range(max_length):
        all_candidates = []  # List to store all potential sequences at this step
        completed = True     # Flag to check if all sequences have ended

        # Loop over all sequences in the current beam
        for score, seq in sequences:
            # If the last token is EOS_TOKEN or max_length is reached, keep the sequence as is
            if seq[-1] == 0 or len(seq) >= max_length:
                all_candidates.append((score, seq))
                continue  # Skip to the next sequence

            completed = False  # At least one sequence can still be expanded

            # Get the probabilities of the next tokens from the model
            probs = model.next_token_proba(seq)

            # Select the top k token probabilities and their corresponding indices
            top_k = min(k, len(probs))  # Adjust k if it's larger than the vocabulary size
            top_k_probs, top_k_tokens = torch.topk(probs, top_k)

            # Create new candidate sequences by appending each of the top k tokens
            for token, prob in zip(top_k_tokens, top_k_probs):
                token = token.item()         # Index of the next token
                prob = prob.item()           # Probability of the next token
                prob = max(prob, epsilon)    # Ensure probability is not zero

                # Calculate the new score by adding the negative log probability
                candidate_score = score - torch.log(torch.tensor(prob))

                # Create a new sequence by appending the token to the existing sequence
                candidate_seq = seq + [token]

                # Add the new candidate sequence and its score to the list of all candidates
                all_candidates.append((candidate_score.item(), candidate_seq))

        # If all sequences have ended, exit the loop early to save computation
        if completed:
            break

        # Sort all candidate sequences by their scores in ascending order (lower score is better)
        all_candidates.sort(key=lambda x: x[0])
        print(_, len(all_candidates))
        print(all_candidates)

        # Select the top k sequences with the best scores to form the new beam
        sequences = all_candidates[:k]

    # Return the sequence with the best score (first in the sorted list)
    # return sequences[0][1]
    return sequences


def beam_search_with_length_norm_and_diversity(model, start_token, k, max_length, alpha, diversity_groups, top_p):
    """
    Perform beam search with length normalization, diverse beam search, and top-p sampling to generate sequences.

    Args:
        model: An object with a method next_token_proba(sequence) -> torch.Tensor.
        start_token (int): The index of the start token in the vocabulary.
        k (int): Beam width, the number of sequences to keep at each step.
        max_length (int): The maximum length of the sequence to generate.
        alpha (float): Length normalization parameter (0 < alpha <= 1).
        diversity_groups (int): Number of diverse groups for Diverse Beam Search.
        top_p (float): Cumulative probability threshold for Top-p (nucleus) sampling.

    Returns:
        List[int]: The most probable sequence of token indices.
    """
    # Divide the beam width across diversity groups
    group_width = max(1, k // diversity_groups)
    sequences = [[(0.0, [start_token])] for _ in range(diversity_groups)]  # List of beams for each group

    epsilon = 1e-10  # Small constant to prevent log(0)

    # Iterate up to the maximum sequence length
    for _ in range(max_length):
        all_group_candidates = []
        completed = True

        for group_id in range(diversity_groups):
            group_sequences = sequences[group_id]
            all_candidates = []

            # Loop over all sequences in the current group's beam
            for score, seq in group_sequences:
                # If the last token is EOS_TOKEN or max_length is reached, keep the sequence as is
                if seq[-1] == 0 or len(seq) >= max_length:
                    normalized_score = score / (len(seq) ** alpha)
                    all_candidates.append((normalized_score, seq))
                    continue  # Skip to the next sequence

                completed = False

                # Get the probabilities of the next tokens from the model
                probs = model.next_token_proba(seq)

                # Apply top-p (nucleus) sampling to filter tokens
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                top_p_mask = cumulative_probs <= top_p
                top_p_indices = sorted_indices[top_p_mask]
                top_p_probs = sorted_probs[top_p_mask]

                # If no tokens are below the threshold, take at least one
                if len(top_p_indices) == 0:
                    top_p_indices = sorted_indices[:1]
                    top_p_probs = sorted_probs[:1]

                # Create new candidate sequences by appending each of the top-p tokens
                for i in range(len(top_p_indices)):
                    token = top_p_indices[i].item()  # Index of the next token
                    prob = top_p_probs[i].item()  # Probability of the next token
                    prob = max(prob, epsilon)  # Ensure probability is not zero

                    # Calculate the new score by adding the negative log probability
                    candidate_score = score - torch.log(torch.tensor(prob))

                    # Create a new sequence by appending the token to the existing sequence
                    candidate_seq = seq + [token]

                    # Add the new candidate sequence and its score to the list of all candidates
                    all_candidates.append((candidate_score.item(), candidate_seq))

            # Sort all candidate sequences by their scores in ascending order (lower score is better)
            all_candidates.sort(key=lambda x: x[0])

            # Select the top sequences for this group
            sequences[group_id] = all_candidates[:group_width]

            # Collect candidates for all groups
            all_group_candidates.extend(all_candidates[:group_width])

        # If all sequences have ended, exit the loop early to save computation
        if completed:
            break

    # Flatten and sort all group candidates to get the best sequences
    all_group_candidates.sort(key=lambda x: x[0])

    # Return the top sequence from all groups
    return all_group_candidates# [0][1]


if __name__ == "__main__":
    model = DummyModel()
    start_token = 1
    k = 3
    max_length = 5

    sequence_indices = beam_search(model, start_token, k, max_length)
    for score, indices in sequence_indices:
        sequence_words = [VOCAB[idx] for idx in indices if idx < len(VOCAB)]
        print(f'score: {score:.2f}, answer: {sequence_words}')