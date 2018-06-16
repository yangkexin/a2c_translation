from tempfile import NamedTemporaryFile
from os import system


def bleu(reference, output):
    with NamedTemporaryFile('w+t', delete=False) as rf, NamedTemporaryFile('w+t', delete=False) as of:
        rf.write('\n'.join(reference))
        of.write('\n'.join(output))
    bleu_f(rf.name, of.name)


def bleu_f(rf, of):
    system('./multi-bleu.perl {} < {}'.format(rf, of))


if __name__ == '__main__':
    reference = 'Two person be in a small race car drive by a green hill .'
    output = 'Two person be in race uniform in a street car .'
    bleu([reference], [output])
