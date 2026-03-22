import json
import random
from pathlib import Path

ALIAS_MAP = {
    'http_post':   ['send_request', 'post_data', 'upload_payload', 'transmit_data'],
    'http_get':    ['fetch_url', 'retrieve_data', 'get_resource'],
    'read_env':    ['get_secret', 'fetch_config', 'load_credential', 'retrieve_key'],
    'read_file':   ['open_file', 'load_file', 'fetch_file'],
    'encode':      ['transform', 'serialize', 'compress'],
    'exec_shell':  ['run_command', 'invoke_process', 'shell_exec'],
    'send_email':  ['mail_data', 'dispatch_message'],
}

# Steps that can be reordered without changing the label
REORDERABLE_PAIRS = [
    ('read_env', 'encode'),
    ('read_file', 'encode'),
    ('read_env', 'read_file'),
]


def apply_alias(text: str) -> str:
    for canonical, aliases in ALIAS_MAP.items():
        alias = random.choice(aliases)
        text = text.replace(f"{canonical}(", f"{alias}(")
    return text


def apply_reorder(text: str) -> str:
    lines = text.split('\n')
    step_lines = [(i, l) for i, l in enumerate(lines) if l.strip().startswith('Step')]
    if len(step_lines) >= 2:
        i1, l1 = step_lines[0]
        i2, l2 = step_lines[1]
        for pair in REORDERABLE_PAIRS:
            if pair[0] in l1 and pair[1] in l2:
                lines[i1] = l1.replace('Step 1', 'Step 1').replace(
                    l1.split(': ', 1)[1], l2.split(': ', 1)[1])
                lines[i2] = l2.replace('Step 2', 'Step 2').replace(
                    l2.split(': ', 1)[1], l1.split(': ', 1)[1])
                break
    return '\n'.join(lines)


def augment(input_path: str, output_path: str, multiplier: int = 2):
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    augmented = list(records)  # keep originals

    for record in records:
        for _ in range(multiplier - 1):
            new_record = dict(record)
            new_input = record['input']

            # Randomly apply one or both transforms
            if random.random() > 0.5:
                new_input = apply_alias(new_input)
            if random.random() > 0.5:
                new_input = apply_reorder(new_input)

            new_record['input'] = new_input
            new_record['augmented'] = True
            augmented.append(new_record)

    random.shuffle(augmented)
    with open(output_path, 'w') as f:
        for r in augmented:
            f.write(json.dumps(r) + '\n')

    print(f"Augmented {len(records)} -> {len(augmented)} examples")


if __name__ == "__main__":
    augment("data/processed/train.jsonl", "data/processed/train_augmented.jsonl", multiplier=2)
