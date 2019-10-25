def represent_in_base(x, b, precision=20):
    # TODO: fix 0 and other things
    result = ''
    i = 0
    while x >= b:
        x /= b
        i += 1
    for i in range(-i, precision):
        if i == 1:
            result += '.'
        if i > 0 and x == 0:
            break
        x *= b
        numeral, x = divmod(x, b)
        result += str(round(numeral))

    return result
