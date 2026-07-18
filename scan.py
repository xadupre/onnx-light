import re
import subprocess

# Match std::string( ARG ) capturing balanced single-level parens in ARG.
CALL = re.compile(r"std::string\(")


def extract_arg(line, start):
    # start points just after 'std::string('
    depth = 1
    i = start
    while i < len(line) and depth:
        c = line[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return line[start:i], i
        i += 1
    return None, None


def is_candidate(arg):
    a = arg.strip()
    if not a:
        return False
    # skip 2-arg constructors (top-level comma)
    depth = 0
    for c in a:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            return False
    # skip string/char literals and numeric constructions
    if '"' in a or "'" in a:
        return False
    # skip pure-pointer/size terminals that need explicit ctor
    if re.search(r"\.(data|c_str)\(\)\s*$", a):
        return False
    # must look like an identifier / member-access / call expression
    if not re.fullmatch(r"[\w:.\-\>\[\]()* ]+", a):
        return False
    # skip things that are clearly numeric or arithmetic
    if re.fullmatch(r"[\d\s+\-*/]+", a):
        return False
    return True


def main():
    files = subprocess.check_output(
        ["git", "ls-files", "onnx_light", "examples", "unittests"], text=True
    ).splitlines()
    for path in files:
        if not path.endswith((".cc", ".cpp", ".h", ".hpp")):
            continue
        with open(path, encoding="utf-8") as f:
            for lno, line in enumerate(f, 1):
                for m in CALL.finditer(line):
                    arg, _ = extract_arg(line, m.end())
                    if arg is None:
                        continue
                    if is_candidate(arg):
                        print(f"{path}:{lno}: std::string({arg.strip()})")


if __name__ == "__main__":
    main()
