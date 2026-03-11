import os
import re


def remove_comments_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # Remove block comments (/* ... */)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # Remove line comments (//...)
    code = re.sub(r'//.*', '', code)

    # Remove trailing whitespace from each line
    code = '\n'.join(line.rstrip() for line in code.splitlines())

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)


def remove_comments_in_directory(directory):
    os.makedirs(directory, exist_ok=True)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.h') or file.endswith('.hpp') or file.endswith('.cu'):
                file_path = os.path.join(root, file)
                print(f'Removing comments from: {file_path}')
                remove_comments_from_file(file_path)


if __name__ == "__main__":
    remove_comments_in_directory('../temp')