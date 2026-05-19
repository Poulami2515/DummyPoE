import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()

# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(question, options, masked_options=None):

    if masked_options is None:
        masked_options = []

    prompt = f"Question:\n{question}\n\nOptions:\n"

    for key, value in options.items():

        if key in masked_options:
            prompt += f"{key}. [MASK]\n"
        else:
            prompt += f"{key}. {value}\n"

    if len(masked_options) > 0:

        masked_str = ", ".join(masked_options)

        prompt += (
            f"\nThe following options are MASKED "
            f"and should be ignored: {masked_str}.\n"
        )

    prompt += "\nAnswer:"

    return prompt

# ============================================================
# COMPUTE:
# log P(y_i | x, Y)
# ============================================================

def compute_logprob(prompt, target_key):

    target_text = f" {target_key}"

    # tokenize prompt
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(model.device)

    # tokenize target
    target_ids = tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(model.device)

    # concatenate
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)

    labels = input_ids.clone()

    # ignore prompt tokens
    labels[:, :prompt_ids.shape[1]] = -100

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            labels=labels
        )

    mean_loss = outputs.loss

    target_len = target_ids.shape[1]

    total_logprob = -mean_loss.item() * target_len

    avg_logprob = -mean_loss.item()

    return total_logprob, avg_logprob

# ============================================================
# SCORE OPTIONS
# ============================================================

def score_options(question, options, masked_options=None):

    prompt = build_prompt(
        question,
        options,
        masked_options
    )

    scores = {}

    for key in options.keys():

        # skip masked options
        if masked_options and key in masked_options:
            continue

        total_ll, avg_ll = compute_logprob(
            prompt,
            key
        )

        scores[key] = {
            "total_logprob": total_ll,
            "avg_logprob": avg_ll
        }

    return scores, prompt

# ============================================================
# STORE BELOW-AVERAGE OPTIONS
# ============================================================

def get_below_average_options(scores):

    avg_score = sum(
        s["avg_logprob"]
        for s in scores.values()
    ) / len(scores)

    low_scoring = []

    for key, value in scores.items():

        if value["avg_logprob"] < avg_score:
            low_scoring.append(key)

    return low_scoring, avg_score

# ============================================================
# RANK OPTIONS
# ============================================================

def rank_options(scores):

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["avg_logprob"],
        reverse=True
    )

    return ranked

# ============================================================
# FULL ELIMINATION PIPELINE
# ============================================================

def elimination_pipeline(question, options):

    print("\n==============================")
    print("ROUND 1 : INITIAL SCORING")
    print("==============================")

    # --------------------------------------------------------
    # INITIAL SCORING
    # --------------------------------------------------------

    initial_scores, initial_prompt = score_options(
        question,
        options
    )

    print("\nINITIAL PROMPT:\n")
    print(initial_prompt)

    print("\nINITIAL SCORES:\n")

    for key, value in initial_scores.items():

        print(f"{key}: {options[key]}")
        print(f"Avg LogProb: {value['avg_logprob']:.4f}")
        print()

    # --------------------------------------------------------
    # BELOW AVERAGE OPTIONS
    # --------------------------------------------------------

    low_scoring, avg_score = get_below_average_options(
        initial_scores
    )

    print("\nAVERAGE SCORE:", round(avg_score, 4))
    print("LOW SCORING OPTIONS:", low_scoring)

    # --------------------------------------------------------
    # MASK LOW OPTIONS
    # --------------------------------------------------------

    print("\n==============================")
    print("ROUND 2 : MASKED REASONING")
    print("==============================")

    masked_scores, masked_prompt = score_options(
        question,
        options,
        masked_options=low_scoring
    )

    print("\nMASKED PROMPT:\n")
    print(masked_prompt)

    print("\nRECALCULATED SCORES:\n")

    for key, value in masked_scores.items():

        print(f"{key}: {options[key]}")
        print(f"Avg LogProb: {value['avg_logprob']:.4f}")
        print()

    # --------------------------------------------------------
    # FINAL RANKING
    # --------------------------------------------------------

    ranked = rank_options(masked_scores)

    print("\n==============================")
    print("FINAL RANKING")
    print("==============================\n")

    for idx, (key, value) in enumerate(ranked):

        print(
            f"{idx+1}. {key} "
            f"(Avg LogProb = "
            f"{value['avg_logprob']:.4f})"
        )

    best_option = ranked[0][0]

    print("\n================================")
    print("FINAL PREDICTION:", best_option)
    print("================================")

    return {
        "initial_scores": initial_scores,
        "masked_scores": masked_scores,
        "masked_options": low_scoring,
        "best_option": best_option
    }

# ============================================================
# EXAMPLE
# ============================================================

question = "Which planet is known as the Red Planet?"

options = {
    "A": "Mars",
    "B": "Jupiter",
    "C": "Venus",
    "D": "Banana"
}

results = elimination_pipeline(
    question,
    options
)