import json
from collections import Counter
from datetime import datetime

def write_character_stats(bios):
    chars = Counter()
    for b in bios:
        for c in b:
            chars[c] += 1
    
    srted_chars = sorted(chars.items(), key=lambda t: -t[1])

    # write out character stats
    outfile = "charstats.txt"
    with open(outfile, 'w') as f:
        f.write(str(datetime.now()) + '\n')
        f.write(f"Source: {src}\n")
        f.write(f"Bio count: {len(bios)}\n")
        f.write(f"Num unique chars: {len(chars)}\n")
        f.write(f"Total tokens: {sum(chars.values())}\n")
        f.write(f"Char, count-----\n")
        for c, ct in srted_chars:
            f.write(f"{c}, {ct}\n")

def write_byte_stats(bios):

    bts = Counter()
    for b in bios:
        for c in b:
            for bt in c.encode():
                bts[bt] += 1
    
    srted_bytes = sorted(bts.items(), key=lambda t: -t[1])

    # write out character stats
    outfile = "byte_stats.txt"
    with open(outfile, 'w') as f:
        f.write(str(datetime.now()) + '\n')
        f.write(f"Source: {src}\n")
        f.write(f"Bio count: {len(bios)}\n")
        f.write(f"Num unique bytes: {len(bts)}\n")
        f.write(f"Total tokens: {sum(bts.values())}\n")
        f.write(f"Byte, count-----\n")
        for b, ct in srted_bytes:
            f.write(f"{b} (0x{b:x}), {ct}\n")

    # also save the set of relevant values
    bytefile = "byte_values.txt"
    with open(bytefile, 'w') as f:
        f.write(', '.join(str(b[0]) for b in srted_bytes))

if __name__ == "__main__":
    src = "bios.json"
    # with open(src) as f:
    #     bios = json.load(f)
    # write_byte_stats(list(bios.values()))
    # ds = StringDataset(src)