import typing as T


def arg_type(name, value, type_):
    if not isinstance(value, type_):
        raise TypeError(f"The type of the argument {name} is {type(value).__qualname__}, but should"
                        f"be {getattr(type_, '__qualname__', type_)}.")


def assert_value_in(name, value, values: T.Collection):
    if value not in values:
        raise TypeError(f"{name} should have a value from {values}, not {value}.")
