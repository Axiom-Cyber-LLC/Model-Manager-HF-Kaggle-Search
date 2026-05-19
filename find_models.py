import os
from pathlib import Path

ROOTS = [
    Path('<Your Model Directory>'),
    Path('<Your Model Directory>'),
    Path('<REDACTED_PATH>'),
    Path('<REDACTED_PATH>'),
    Path('<REDACTED_PATH> Support/Apollo/models'),
    Path('<REDACTED_PATH> Support/OffGrid/models'),
    Path('<REDACTED_PATH> Support/LocallyAI/models'),
    Path('<REDACTED_PATH> Support/nomic.ai/GPT4All'),
]
EXTS = {'.gguf', '.safetensors', '.bin', '.pt', '.pth', '.mlmodel', '.mlpackage', '.onnx'}
MIN_SIZE = 50 * 1024 * 1024

# Group by (dev, inode) so hardlinks/dupes collapse — but keep ALL paths per blob
seen = {}
for root in ROOTS:
    if not root.is_dir(): continue
    for p in root.rglob('*'):
        try:
            if not (p.is_file() and not p.is_symlink()):
                continue
            if p.suffix.lower() not in EXTS:
                continue
            st = p.stat()
            if st.st_size < MIN_SIZE:
                continue
            key = (st.st_dev, st.st_ino)
            seen.setdefault(key, [st.st_size, []])
            seen[key][1].append(str(p))
        except OSError:
            continue

# Also include Ollama blobs (no extension; sha256-named)
ollama_blobs = Path.home() / '.ollama' / 'models' / 'blobs'
if ollama_blobs.is_dir():
    for p in ollama_blobs.iterdir():
        try:
            if not p.is_file(): continue
            st = p.stat()
            if st.st_size < MIN_SIZE: continue
            key = (st.st_dev, st.st_ino)
            seen.setdefault(key, [st.st_size, []])
            seen[key][1].append(str(p))
        except OSError:
            continue

# Sort by size desc
rows = sorted(seen.values(), key=lambda r: -r[0])
total = sum(r[0] for r in rows)

# Write to file AND stdout
out_path = Path.home() / 'model_inventory.txt'
with out_path.open('w') as f:
    header = f'{len(rows)} unique model blobs, total {total/1024**4:.2f} TB ({total/1024**3:.0f} GB)'
    print(header)
    f.write(header + '\n\n')
    print('=' * 100)
    f.write('=' * 100 + '\n\n')
    for size, paths in rows:
        sz = f'{size/1024**3:.2f} GB' if size >= 1024**3 else f'{size/1024**2:.0f} MB'
        # Print first path as 'main', remaining as additional
        for i, p in enumerate(sorted(paths)):
            tag = sz if i == 0 else ' ' * len(sz)
            line = f'{tag:>10}  {p}'
            print(line)
            f.write(line + '\n')
        if len(paths) > 1:
            print()
            f.write('\n')

print()
print(f'Full list also written to: {out_path}')
