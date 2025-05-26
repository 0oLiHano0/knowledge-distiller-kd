"""
relative_to_absolute.py

将指定目录下所有 .py 文件中的相对导入改为绝对导入
Usage:
    python relative_to_absolute.py <root_dir> [--package <root_pkg>] [--dry-run]
说明:
    <root_dir>  : 该目录被视为某个 Python 顶层包的根
    --package   : 顶层包名；若不指定，则自动使用 <root_dir> 文件夹名
    --dry-run   : 仅打印转换结果，不写回文件
"""
import ast
import argparse
import pathlib
import sys
from typing import Optional
import astor


class RelImportRewriter(ast.NodeTransformer):
    """把 ImportFrom 中的相对导入(level>0) 改为绝对导入"""

    def __init__(self, module_path: str, root_pkg: str):
        super().__init__()
        self.module_path_parts = module_path.split('.')
        self.root_pkg = root_pkg

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level and node.level > 0:
            if node.level > len(self.module_path_parts):
                return node
            ancestor_parts = self.module_path_parts[:-node.level]
            base = '.'.join([self.root_pkg, *ancestor_parts]
                ) if ancestor_parts else self.root_pkg
            new_module = f'{base}.{node.module}' if node.module else base
            return ast.ImportFrom(module=new_module, names=node.names,
                level=0, lineno=node.lineno, col_offset=node.col_offset)
        return node


def absolute_module_path(file_path: pathlib.Path, root_dir: pathlib.Path
    ) ->str:
    """ /root/pkg/sub/mod.py -> sub.mod """
    rel = file_path.relative_to(root_dir).with_suffix('')
    return '.'.join(rel.parts)


def process_file(py_file: pathlib.Path, root_dir: pathlib.Path, root_pkg:
    str, dry_run: bool=False) ->bool:
    code = py_file.read_text(encoding='utf-8')
    tree = ast.parse(code)
    module_path = absolute_module_path(py_file, root_dir)
    rewriter = RelImportRewriter(module_path, root_pkg)
    new_tree = rewriter.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_code = astor.to_source(new_tree)
    if new_code != code:
        if dry_run:
            print(f'\n==== {py_file} ====')
            print(new_code)
        else:
            py_file.write_text(new_code, encoding='utf-8')
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=
        'Convert relative imports to absolute imports')
    parser.add_argument('root_dir', help='目录被视为顶层包根')
    parser.add_argument('--package', help='顶层包名，默认 = 根目录名')
    parser.add_argument('--dry-run', action='store_true', help='只打印结果，不写回文件')
    args = parser.parse_args()
    root_dir = pathlib.Path(args.root_dir).resolve()
    if not root_dir.is_dir():
        sys.exit('root_dir 不是有效目录')
    root_pkg = args.package or root_dir.name
    changed = 0
    for py_file in root_dir.rglob('*.py'):
        changed += process_file(py_file, root_dir, root_pkg, args.dry_run)
    print(f'\nDone. {changed} file(s) modified.')


if __name__ == '__main__':
    main()
