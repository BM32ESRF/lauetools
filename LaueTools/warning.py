"""
module to warn user that function are deprecated

# === Examples of use ===

@deprecated
def some_old_function(x,y):
    return x + y

class SomeClass:
    @deprecated
    def some_old_method(self, x,y):
        return x + y

"""

import warnings
import functools


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        """
        new_func
        """
        warnings.warn_explicit(
            "Call to deprecated function %(funcname)s." % {"funcname": func.__name__},
            category=DeprecationWarning,
            filename=func.__code__.co_filename,
            lineno=func.__code__.co_firstlineno + 1,
        )

        print("\n\n*********** LAUETOOLS warning *************        ")
        print("Call to deprecated function %(funcname)s" % {"funcname": func.__name__})
        print(
            "in module %(filename)s at linenumber %(linenumber)d."
            % {
                "filename": func.__code__.co_filename,
                "linenumber": func.__code__.co_firstlineno + 1,
            }
        )
        print("*" * 43 + "\n\n")
        return func(*args, **kwargs)

    return new_func


@deprecated
def some_old_function(x, y):
    """
    example
    """
    return x + y
