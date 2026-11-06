"""Deterministic documentation and Python AST checks."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
from .phrases import LIPREADING, NOLIPREADING, GENERIC_NAMES


def match_phrases(text, phrases):
    text = text.lower()
    found = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = text.find(phrase)
        while start >= 0:
            found.append(phrase)
            text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
            start = text.find(phrase)
    return found, text


def prose(text):
    lines = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            fenced = not fenced
        elif not fenced:
            lines.append(line.replace('`', ''))
    return '\n'.join(lines)


def evidence(source):
    tree = ast.parse(source)
    aliases = {}
    shadows = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                aliases[a.asname or a.name] = (node.module or '') + '.' + a.name

    def resolve(node):
        if isinstance(node, ast.Name):
            return '' if node.id in shadows else aliases.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base and node.attr not in shadows else ''
        return ''

    def av(node):
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and 'av-hubert' in node.value.lower()

    def construction(node):
        if not isinstance(node, ast.Call):
            return None
        name = resolve(node.func)
        if name == 'transformers.AutoModelForCTC.from_pretrained' and node.args and av(node.args[0]):
            return 'av'
        if name.endswith('.LipNet') and 'lipnet' in name.lower():
            return 'lip'
        return None

    instances = {}
    # Resolve assignments to a fixed point, including simple instance aliases.
    for _ in range(len(list(ast.walk(tree)))):
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                kind = construction(value) or (instances.get(value.id) if isinstance(value, ast.Name) else None)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if kind and isinstance(target, ast.Name) and target.id not in shadows and instances.get(target.id) != kind:
                        instances[target.id] = kind
                        changed = True
        if not changed:
            break
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = resolve(node.func)
        if name == 'transformers.pipeline' and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'automatic-speech-recognition' and any(k.arg == 'model' and av(k.value) for k in node.keywords):
            return True
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in GENERIC_NAMES and attr not in shadows:
                return True
            base = node.func.value
            kind = construction(base) or (instances.get(base.id) if isinstance(base, ast.Name) else None)
            if (kind == 'av' and attr == 'generate') or (kind == 'lip' and attr in ('predict', 'predict_classes')):
                return True
        elif isinstance(node.func, ast.Name):
            if name and name.rsplit('.', 1)[-1] in GENERIC_NAMES:
                return True
            if instances.get(node.func.id) == 'av':
                return True
        elif construction(node.func) == 'av':
            return True
    return False


def audit(repo, ignores=(), verbose=False):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository is not a directory: ' + str(root))
    docs = []
    has_evidence = False
    excluded = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env'}
    decisions = []
    def walk(directory):
        for path in sorted(directory.iterdir()):
            rel = path.relative_to(root).as_posix()
            if any(rel.startswith(prefix) for prefix in ignores) or path.name in excluded or path.is_symlink():
                decisions.append('skip ' + rel)
                continue
            if path.is_dir():
                if (path / 'pyvenv.cfg').exists():
                    decisions.append('skip ' + rel)
                else:
                    yield from walk(path)
            else:
                yield path, rel
    for path, rel in walk(root):
        if path.suffix == '.py':
            try:
                present = evidence(path.read_text(encoding='utf-8'))
            except SyntaxError as exc:
                raise OSError('Cannot parse ' + rel + ': ' + str(exc)) from exc
            has_evidence = has_evidence or present
            decisions.append('source ' + rel + ': ' + ('evidence' if present else 'no evidence'))
        if rel == 'README.md' or (rel.startswith('docs/') and path.suffix == '.md') or ('tests' in path.parts and path.suffix in ('.md', '.py', '.txt')):
            text = prose(path.read_text(encoding='utf-8'))
            negatives, remaining = match_phrases(text, NOLIPREADING)
            positives, _ = match_phrases(remaining, LIPREADING)
            docs.append((rel, positives, negatives, rel != 'README.md' and not rel.startswith('docs/')))
            decisions.append('documentation ' + rel + ': %d positive, %d denial' % (len(positives), len(negatives)))
    findings = []
    shipped_positive = any(pos for _, pos, _, test in docs if not test)
    for rel, positives, negatives, test in docs:
        if positives and not has_evidence and (not test or not shipped_positive):
            findings.append(dict(id='documented-lip-reading-claim-false', severity='MEDIUM' if test else 'HIGH', verdict='FAIL', message='Documented checked capability has no Python evidence.', path=rel, phrases=positives))
        if negatives and has_evidence and not test:
            findings.append(dict(id='documented-no-lip-reading-claim-false', severity='HIGH', verdict='FAIL', message='Documented denial contradicts Python evidence.', path=rel, phrases=negatives))
    return findings, decisions


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, message + '\n')


def main(argv=None):
    parser = Parser(description='Audit documentation claims against static Python evidence.')
    parser.add_argument('--version', action='version', version='0.1.0')
    sub = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    cmd = sub.add_parser('audit')
    cmd.add_argument('repo', nargs='?', default='.')
    cmd.add_argument('--json', action='store_true')
    cmd.add_argument('--verbose', action='store_true')
    cmd.add_argument('--ignore', action='append', nargs='+', default=[])
    cmd.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    args = parser.parse_args(argv)
    try:
        findings, decisions = audit(args.repo, [p for group in args.ignore for p in group])
    except (OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.verbose:
        print('Scope: README.md, docs/**/*.md, tests fixtures; Python sources in ' + str(Path(args.repo).resolve()), file=sys.stderr)
        print('\n'.join(decisions), file=sys.stderr)
    if args.json:
        print(json.dumps(findings))
    elif findings:
        for finding in findings:
            print('{severity} {id} {path}: {message}'.format(**finding))
    else:
        print('PASS')
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0
