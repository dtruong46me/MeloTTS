import os

fixes = {
    r'd:\Projects\TEST\MeloTTS\melo\text\fr\core.py': [
        ('from . import symbols',                'from .. import symbols'),
        ('from .fr_phonemizer import cleaner',   'from .phonemizer import cleaner'),
        ('from .fr_phonemizer import fr_to_ipa', 'from .phonemizer import fr_to_ipa'),
    ],
    r'd:\Projects\TEST\MeloTTS\melo\text\es\core.py': [
        ('from . import symbols',                'from .. import symbols'),
        ('from .es_phonemizer import cleaner',   'from .phonemizer import cleaner'),
        ('from .es_phonemizer import es_to_ipa', 'from .phonemizer import es_to_ipa'),
    ],
    r'd:\Projects\TEST\MeloTTS\melo\text\jp\core.py': [
        ('from . import symbols', 'from .. import symbols'),
    ],
    r'd:\Projects\TEST\MeloTTS\melo\text\kr\core.py': [
        ('from . import punctuation, symbols', 'from .. import punctuation, symbols'),
        ('from . import japanese_bert',         'from ..jp import bert as japanese_bert'),
    ],
}

for filepath, changes in fixes.items():
    if not os.path.exists(filepath):
        print('MISSING:', filepath)
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content
    for old, new in changes:
        content = content.replace(old, new)
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print('Fixed:', os.path.basename(filepath))
    else:
        print('No changes:', os.path.basename(filepath))

print('All done.')
