from functools import wraps


def damped(f, beta: float = 1.0):
    """
    Returns a damped version of f, i.e. (1-β)*x + β*f(x)
    """

    def damped_f(Ux, z, *args, **kwargs):
        return (1 - beta) * z + beta * f(Ux, z, *args, **kwargs)

    return damped_f


def track_states(f, states: list, disable: bool = False):
    """
    Returns decorated f so that the result of every call is stored in the 'states' list.
    If disable=True, we undo this decoration.
    """
    if disable:
        return f.__wrapped__
    else:

        def tracking_decorator(f):
            @wraps(f)
            def tracking_wrapper(*args, **kwargs):
                result = f(*args, **kwargs)
                states.append(result.detach())
                return result

            return tracking_wrapper

        return tracking_decorator(f)
