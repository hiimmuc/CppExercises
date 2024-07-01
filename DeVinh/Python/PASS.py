import math
import time
import numpy as np


def longestUniqueSubStr(string):

    # last index of every character
    last_idx = {}
    max_len = 0

    # starting index of current
    # window to calculate max_len
    start_idx = 0
    #
    sub_str_len_map = {}

    for i in range(0, len(string)):

        # Find the last index of str[i]
        # Update start_idx (starting index of current window)
        # as maximum of current value of start_idx and last
        # index plus 1
        if string[i] in last_idx:
            start_idx = max(start_idx, last_idx[string[i]] + 1)

        # Update result if we get a larger window
        max_len = max(max_len, i-start_idx + 1)

        # Update last index of current char.
        last_idx[string[i]] = i

        #
        current_str = string[start_idx:start_idx+max_len]
        sub_str_len_map.setdefault(len(current_str), []).append(
            current_str)

    sub_str_len_map = sorted(sub_str_len_map.items())

    # return longest, latest substring
    return sub_str_len_map[-1][-1][-1]


def get_pw_instr(string):
    words = [word.strip() for word in string.split(" ")]
    password_in_word = [longestUniqueSubStr(word) for word in words]
    password_in_word.sort(key=lambda x: len(x))
    return password_in_word[-1]


def get_password(n: int, sequences: list):
    password_in_string = [get_pw_instr(sequences[i]) for i in range(n)]
    password_in_string.sort(key=lambda x: len(x))
    return password_in_string[-1]


if __name__ == "__main__":
    with open('./INP/PASS.INP') as f:
        data = [line.replace('\n', '') for line in f.readlines()]

    n = data[0]
    sequences = data[1:]

    result = get_password(int(n), sequences)
    print(result)

    with open("./OUT/PASS.OUT", 'w') as wf:
        wf.write(str(result))
