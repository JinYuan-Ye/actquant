"""Extract bundled source without overwriting any existing files."""
from pathlib import Path
import zipfile, hashlib
root = Path(__file__).resolve().parent.parent
vendor = root / 'src/ActQuant/vendor'
with zipfile.ZipFile(vendor / 'tokenizers-cpp.zip') as z:
    for info in z.infolist():
        dest = vendor / info.filename
        if not dest.resolve().is_relative_to(vendor.resolve()):
            raise RuntimeError(f'Unsafe archive path: {info.filename}')
        if info.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif dest.exists():
            if dest.read_bytes() != z.read(info):
                raise RuntimeError(f'Refusing to overwrite: {dest}')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))
print('Bundled tokenizer source ready; existing files preserved.')
