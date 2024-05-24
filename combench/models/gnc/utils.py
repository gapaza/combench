



# function to enumerate all possible binary strings of length n
def enumerate_binary_strings(n):
    """
        Generate all binary strings of length n.

        :param n: Length of the binary strings.
        :return: List of binary strings of length n.
        """
    if n < 1:
        return []

    result = []

    def backtrack(current):
        if len(current) == n:
            result.append(current)
            return
        backtrack(current + '0')
        backtrack(current + '1')

    backtrack('')
    return result



if __name__ == '__main__':

    # Test the function
    n = 10
    binary_strings = enumerate_binary_strings(n)
    print(len(binary_strings))
    # Output: ['000', '001', '010', '011', '100', '101', '110', '111']






