import os


def replace_string_in_file(file_path, old_string, new_string):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace the old string with the new string
    content = content.replace(old_string, new_string)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def replace_string_in_directory(directory, old_string, new_string):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.h'):
                file_path = os.path.join(root, file)
                print(f'Replacing "{old_string}" with "{new_string}" in: {file_path}')
                replace_string_in_file(file_path, old_string, new_string)


if __name__ == "__main__":
    old_str = "<atlas/core/macros.h>"  # Replace with the string you want to replace
    new_str = ""  # Replace with the new string
    replace_string_in_directory("../include", old_str, new_str)